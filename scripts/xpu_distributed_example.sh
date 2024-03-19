#!/bin/bash
#SBATCH --partition=pvc
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=10
#SBATCH --gpus-per-node=10
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=1:00:00

# determine number of processes and processes per node
# from SLURM control variables
export NP=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
export PPN=$SLURM_NTASKS_PER_NODE

# write nodefile to pass to MPI
scontrol show hostnames > nodefile.${SLURM_JOBID}

# map-by binds processes according to their CPU socket
# nodefile informs MPI which nodes to run on
# bootstrap ssh is required to override SLURM control
mpirun -n $NP -ppn $PPN -map-by numa \
    -f nodefile.${SLURM_JOBID} -genvall \
    -bootstrap ssh \
    python mace/scripts/run_train.py \
    --name='model' \
    --model='MACE' \
    --num_interactions=2 \
    --num_channels=128 \
    --max_L=2 \
    --correlation=3 \
    --E0s='average' \
    --r_max=5.0 \
    --train_file='./h5_data/train.h5' \
    --valid_file='./h5_data/valid.h5' \
    --statistics_file='./h5_data/statistics.json' \
    --num_workers=8 \
    --batch_size=20 \
    --valid_batch_size=80 \
    --max_num_epochs=100 \
    --loss='weighted' \
    --error_table='PerAtomRMSE' \
    --default_dtype='float32' \
    --device='xpu' \
    --distributed \
    --seed=2222 \
