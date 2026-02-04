#!/bin/bash
#SBATCH --job-name=eval_msgym
#SBATCH --output=logs/eval_msgym_job%j.log
#SBATCH --error=logs/eval_msgym_job%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=P100
#SBATCH --gpus=1
#SBATCH --chdir=/home/ids/silva-21/GraphConf/Models/SpecBridge


# Run evaluation
python -m specbridge.eval.candidates \
    --mgf data/SpecBridge_MassSpecGym_dataset.mgf \
    --dreams-ckpt runs/DreaMS/ssl_model.ckpt \
    --adapter-ckpt runs/msgym/SpecBridge_MSGYM_checkpoint.pt \
    --candidates data/SpecBridge_MSGYM_candidates.pkl \
    --fold-query test \
    --use-mapped \
    --deterministic-map \
    --no-gaussian \
    --batch-size 32 \
    --cond-dim 2048 \
    --mapper-hidden 2048 \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m \
    --cache-cand-emb cache/msgym_test_chemberta.pt

echo "Evaluation completed!"
