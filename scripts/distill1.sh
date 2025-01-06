python distill-sim.py --gpu 1 --init_pair summary --seed 42 --dataset_name photo
python distill-sim.py --gpu 1 --init_pair summary --seed 43 --dataset_name photo
python distill-sim.py --gpu 1 --init_pair summary --seed 44 --dataset_name photo

python distill-sim.py --gpu 0 --init_pair random --seed 42 --dataset_name computer
python distill-sim.py --gpu 0 --init_pair random --seed 43 --dataset_name computer
python distill-sim.py --gpu 0 --init_pair random --seed 44 --dataset_name computer