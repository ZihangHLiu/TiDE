#!/bin/bash
#SBATCH --array=70-71
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=pavia
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:2
#SBATCH -t 1-24:00 # time requested (D-HH:MM)

pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate lzhenv
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

root=
cd ${root}

print_tofile=True
datadir=${root}/data
cuda=True
dataset=ETTh1
epoch=10
batch_size=8
cuda=True
lr=3.82e-5
ckpt_path=/scratch/zhliu/checkpoints/TiDE/epoch_${epoch}/batch_size_${batch_size}/lr_${lr}
save_path=${ckpt_path}

mkdir -p ${ckpt_path}

cd src
pwd
CUDA_VISIBLE_DEVICES=0,1,2,3  python train.py \
    --print-tofile ${print_tofile} \
    --ckpt_path ${ckpt_path} \
    --datadir ${datadir} \
    --dataset ${dataset} \
    --save-path ${save_path} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --cuda ${cuda} \
    --lr ${lr} \
