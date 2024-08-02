#!/bin/bash -l
#SBATCH --job-name=train_mg_ds
#SBATCH --partition=standard-g
#SBATCH --time=01:00:00

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G

#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=project_462000008
#SBATCH --exclusive

ml LUMI/23.03 partition/G
ml PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240404

# Hostfile
###############################################################################
# nodelist=$(scontrol show hostname $SLURM_NODELIST)
# printf "%s\n" "${nodelist[@]}" > myhostfile
# 
# function makehostfile() {
# perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
# $slots=8 if $slots==0; # workaround 8 gpu machines
# @nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
# print map { "$b$_ slots=$slots\n" } @nodes'
# }
# makehostfile > myhostfile
###############################################################################

srun singularity exec $SIF bash -c 'export CXX=g++-10; export home_dir=/users/hdreuning; cd $home_dir; export BASE_SRC_PATH=$home_dir/Megatron-DeepSpeed; export BASE_DATA_PATH=${BASE_SRC_PATH}/dataset; cd $home_dir/Megatron-DeepSpeed/examples_deepspeed/rebase; bash lumi_ds_pretrain_gpt_13b.sh'
srun singularity exec $SIF bash -c 'export CXX=g++-10; export home_dir=/users/username; cd $home_dir; export BASE_SRC_PATH=$home_dir/Megatron-DeepSpeed; export BASE_DATA_PATH=${BASE_SRC_PATH}/dataset; cd $home_dir/Megatron-DeepSpeed/examples_deepspeed/rebase; bash lumi_ds_pretrain_gpt_13b.sh'
