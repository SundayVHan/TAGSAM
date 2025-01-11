python distill-mtt.py --gpu 0 --init_pair random --seed 42 --dataset_name computer --num_syn 200
python distill-mtt.py --gpu 0 --init_pair random --seed 43 --dataset_name computer --num_syn 200
python distill-mtt.py --gpu 0 --init_pair random --seed 44 --dataset_name computer --num_syn 200

python distill-mtt.py --gpu 0 --init_pair random --seed 42 --dataset_name photo --num_syn 100
python distill-mtt.py --gpu 0 --init_pair random --seed 43 --dataset_name photo --num_syn 100
python distill-mtt.py --gpu 0 --init_pair random --seed 44 --dataset_name photo --num_syn 100

python distill-mtt.py --gpu 0 --init_pair random --seed 42 --dataset_name computer --num_syn 100
python distill-mtt.py --gpu 0 --init_pair random --seed 43 --dataset_name computer --num_syn 100
python distill-mtt.py --gpu 0 --init_pair random --seed 44 --dataset_name computer --num_syn 100

python distill-mtt.py --gpu 0 --init_pair random --seed 42 --dataset_name arxiv --num_syn 100
python distill-mtt.py --gpu 0 --init_pair random --seed 43 --dataset_name arxiv --num_syn 100
python distill-mtt.py --gpu 0 --init_pair random --seed 44 --dataset_name arxiv --num_syn 100

python distill-mtt.py --gpu 0 --init_pair random --seed 42 --dataset_name arxiv --num_syn 500
python distill-mtt.py --gpu 0 --init_pair random --seed 43 --dataset_name arxiv --num_syn 500
python distill-mtt.py --gpu 0 --init_pair random --seed 44 --dataset_name arxiv --num_syn 500

python distill-sim.py --gpu 0 --init_pair random --seed 42 --dataset_name photo --num_syn 500
python distill-sim.py --gpu 0 --init_pair random --seed 43 --dataset_name photo --num_syn 500
python distill-sim.py --gpu 0 --init_pair random --seed 44 --dataset_name photo --num_syn 500

python distill-sim.py --gpu 0 --init_pair summary --seed 42 --dataset_name computer --num_syn 500
python distill-sim.py --gpu 0 --init_pair summary --seed 43 --dataset_name computer --num_syn 500
python distill-sim.py --gpu 0 --init_pair summary --seed 44 --dataset_name computer --num_syn 500

python distill-sim.py --gpu 0 --init_pair random --seed 42 --dataset_name arxiv --num_syn 500
python distill-sim.py --gpu 0 --init_pair random --seed 43 --dataset_name arxiv --num_syn 500
python distill-sim.py --gpu 0 --init_pair random --seed 44 --dataset_name arxiv --num_syn 500
