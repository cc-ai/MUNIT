#!/bin/bash
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:P6000
#SBATCH --mem=24GB
#SBATCH --gres=gpu:titanx:1
#SBATCH --time=48:00:00
#SBATCH -o /network/juravera/slurmlogs/slurm-%j.out
#SBATCH --job-name jupyter-notebook
#SBATCH --output ./slurnlogs/jupyter-notebook-%J.log
#SBATCH -x kepler5 
# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')
# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@login-1.server.mila.quebec -p 8001    
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH cluster: ${cluster}.login-1.server.mila.quebec
SSH login: $user
SSH port: $port
Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"
# load modules or conda environments here
module load miniconda/3
module load cuda-10.0/cudnn/7.5
source $CONDA_ACTIVATE
conda activate simtoreal
# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
python -m notebook --no-browser --port=${port} --ip=${node}
python -m notebook list
