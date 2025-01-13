from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from data.dataset import read_text_list


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = read_text_list("art")

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v1')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v1')
model.to("cuda")

# Tokenize sentences
batch_size = 32
all_embeddings = []
for i in tqdm(range(0, len(sentences), batch_size)):
    # Select a batch of sentences
    batch = sentences[i:i + batch_size]

    # Tokenize batch
    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda")

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Collect embeddings
    all_embeddings.append(sentence_embeddings.cpu())

# Concatenate all embeddings
all_embeddings = torch.cat(all_embeddings, dim=0)

torch.save(all_embeddings, "data/art/art_embeddings.pt")
