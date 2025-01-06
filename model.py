import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda import graph
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        return x


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args

        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.target_token_idx = 0

        if args.use_text_emb:
            self.requires_grad_(False)

    def forward(self, texts):
        inputs = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True).to(self.args.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, self.target_token_idx, :]


class GraphEncoder(nn.Module):
    def __init__(self, args):
        super(GraphEncoder, self).__init__()
        self.args = args

        if args.graph_encoder == 'gcn':
            self.model = GCN(args.gnn_input_dim, args.gnn_hidden_dim, args.gnn_output_dim)
        else:
            raise ValueError('Invalid graph encoder')

    def forward(self, x, edge_index):
        return self.model(x, edge_index)


class TextProjection(nn.Module):
    def __init__(self, args):
        super(TextProjection, self).__init__()
        self.args = args
        dropout_rate = 0.1

        self.projection = nn.Linear(args.text_emb_dim, args.gnn_output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(args.gnn_output_dim, args.gnn_output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(args.gnn_output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIP(nn.Module):
    def __init__(self, args):
        super(CLIP, self).__init__()
        self.args = args

        if self.args.use_text_emb:
            self.text_encoder = TextProjection(args)
        else:
            self.text_encoder = nn.Sequential(
                TextEncoder(args),
                TextProjection(args)
            )

        self.graph_encoder = GraphEncoder(args)

    def forward(self, node_f, edge_index, node_idx, texts, is_eval=False, is_distill=False):
        text_emb = self.encode_text(texts)
        graph_emb = self.encode_graph(node_f, edge_index, node_idx)

        logits = np.exp(np.log(1 / 0.07)) * graph_emb @ text_emb.t()

        if is_eval:
            if is_distill:
                return logits, graph_emb, text_emb
            else:
                return logits
        else:
            ground_truth = torch.arange(len(logits)).type_as(logits).long()
            loss = (F.cross_entropy(logits, ground_truth) + F.cross_entropy(logits.t(), ground_truth)) / 2
            return loss, logits

    def encode_graph(self, node_f, edge_index, node_idx):
        graph_emb = self.graph_encoder(node_f, edge_index)
        graph_emb = graph_emb[node_idx]
        graph_emb = graph_emb / graph_emb.norm(dim=1, keepdim=True)
        return graph_emb

    def encode_text(self, texts):
        text_emb = self.text_encoder(texts)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
        return text_emb


class PromptLearner(nn.Module):
    def __init__(self, args, classnames, g_texts):
        super().__init__()
        self.vars = nn.ParameterList()
        n_cls = len(classnames)
        n_ctx = args.coop_n_ctx

        text_encoder = TextEncoder(args)

        prompts = text_encoder.tokenizer(g_texts.tolist(), return_tensors="pt", truncation=True, padding=True)["input_ids"]
        with torch.no_grad():
            embedding = text_encoder.model.get_input_embeddings()(prompts)
        ctx_vectors = embedding[:, 1:1 + n_ctx, :]
        ctx_vectors = ctx_vectors.mean(dim=1)

        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.vars.append(self.ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(text_encoder.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = text_encoder.tokenizer(prompts, return_tensors="pt", truncation=True, padding=True)["input_ids"]
        with torch.no_grad():
            embedding = text_encoder.model.get_input_embeddings()(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

    def parameters(self):
        return self.vars


class CoOp(nn.Module):
    """Context Optimization (CoOp).
    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def __init__(self, args, classnames, clip_model, g_texts):
        super().__init__()
        self.args = args
        self.classnames = classnames
        self.text_encoder = TextEncoder(args)
        self.clip = clip_model
        self.prompt_learner = PromptLearner(args, classnames, g_texts)

        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.optim = optim.Adam(self.prompt_learner.parameters(), lr=args.prompt_lr)

    def forward(self, node_f, edge_index, node_idx, label, training=False):
        prompts = self.prompt_learner()
        prompt_emb = self.text_encoder.model(inputs_embeds=prompts).last_hidden_state[:, 0, :]
        graph_emb = self.clip.encode_graph(node_f, edge_index, node_idx)
        text_emb = self.clip.encode_text(prompt_emb)
        logits = np.exp(np.log(1 / 0.07)) * graph_emb @ text_emb.t()

        if training:
            loss = F.cross_entropy(logits, label)
            self.optim.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()
            self.optim.step()

        return logits


class GPT2:
    """
    Citation: https://github.com/HendrikStrobelt/detecting-fake-text/blob/master/backend/api.py
    Model class for GPT-2. Primarily used to obtain word probabilities
    """

    def __init__(self, device="cpu", location=""):
        if location == "":
            self.enc = GPT2Tokenizer.from_pretrained("gpt2-large")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        else:
            self.enc = GPT2Tokenizer.from_pretrained(location)
            self.model = GPT2LMHeadModel.from_pretrained(location)
        self.device = torch.device(device)
        self.model.eval()
        self.start_tok = "<|endoftext|>"
        # SPECIAL_TOKENS = ["<pad>"]
        # self.enc.add_special_tokens(SPECIAL_TOKENS)
        # self.model.set_num_special_tokens(len(SPECIAL_TOKENS))
        self.model.to(self.device)

    def pad(self, context, max_length=1024):
        for i in range(len(context)):
            if len(context[i]) > max_length:
                context[i] = context[i][:max_length]
        max_len = max([len(sentence) for sentence in context])
        # print("Maximum Length: ", max_len)
        for i in range(len(context)):
            # print(len(context[i]), max_len - len(context[i]))
            for j in range(max_len - len(context[i])):
                context[i].append(context[i][0])
            # print(max_len - len(sentences[i].split()))
            # print("i: ", sentences[i])
        # print([[self.enc.encode("<pad>") for idx in range(max_len - len(in_text[i].split()))]  for i in range(len(in_text))])
        # print(sentences)
        # print([len(context[i]) for i in range(len(context))])
        return context

    def get_probabilities(self, in_text, topk=40):
        """
        Take in a sequence of text tokens, make predictions on each word given past context and
        return topk

        Returns:
            Dictionary "payload" containing:
            real_probs
                - List of tuples, one for each token in sequence
                - Probability of the actual words in the sequence
                - Each tuple of the form (position of next word in prediction, predicted probability)

            context_strings:
                - Strings in the sequence along with start token
        """
        with torch.no_grad():
            start_tok = torch.full((1, 1), self.enc.encoder[self.start_tok],
                                   device=self.device, dtype=torch.long)
            context = [self.start_tok + " " + in_text[i] for i in range(len(in_text))]
            context = [self.enc.encode(context[i]) for i in range(len(context))]
            context = self.pad(context)
            context = torch.tensor(context, device=self.device, dtype=torch.long)
            output = self.model(context)
            logits = output.logits
            yhat = torch.softmax(logits[:, :-1], dim=-1)
            y = context[:, 1:]
            real_topk_probs = [yhat[t][np.arange(0, y[t].shape[0], 1), y[t]].data.cpu().numpy().tolist() for t in
                               range(yhat.shape[0])]
            real_topk_probs = [list(map(lambda x: round(x, 15), real_topk_probs[t])) for t in
                               range(len(real_topk_probs))]

            real_topk = [list(real_topk_probs[t]) for t in range(len(real_topk_probs))]

            context_strings = [[self.enc.decoder[s.item()] for s in context[t]] for t in range(len(context))]
            context_strings = [[self.postprocess(s) for s in context_strings[t]] for t in range(len(context_strings))]
            del context, logits, y, yhat,
            torch.cuda.empty_cache()
        """ 
        pred_topk = [[list(zip([self.enc.decoder[p] for p in sorted_preds[t][i][:topk]],
            list(map(lambda x: round(x, 5),yhat[t][i][sorted_preds[t][i][
                :topk]].data.cpu().numpy().tolist()))))
                    for i in range(y[t].shape[0])] for t in range(y.shape[0])]
        pred_topk = [[[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk[t]] for t in range(len(pred_topk))]
        """
        payload = {'context_strings': context_strings,
                   'real_probs': real_topk}  # , 'pred_topk': pred_topk}

        # del context, logits, y, yhat,
        # torch.cuda.empty_cache()
        # code.interact(local=locals())
        return payload

    def postprocess(self, token):
        with_space = False
        with_break = False
        # print(token, token[0], token[1:]),
        if token[0] == 'Ġ':
            with_space = True
            token = token[1:]
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        if len(token) > 0 and token[0] == "Â":
            token = token[1:]
        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token
        # if with_space:
        #    token = '\u0120' + token
        # if with_break:
        #    token = '\u010A' + token
        # print(token)
        return token