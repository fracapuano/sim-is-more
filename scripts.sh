# script to train 
python train.py --n-envs 1 --env marcella-plus --parallel-envs False --use-wandb-callback False
python train.py --n-envs 3 --env marcella-plus --parallel-envs True --use-wandb-callback True