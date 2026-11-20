#!/bin/bash
set -euo pipefail

REF=reference/v1_preprint
BACKBONE=data/high_qual_alignments/backbone.filtered_only.afa
TRAITS=data/high_qual_alignments/temp_data.tsv
for L in 1 6; do
  roseate align -o tests/layer$L --backbone $BACKBONE --train-ids reproduce/corkrey_sequence_ids.txt
  roseate embed -o tests/layer$L --layer $L \
      --context-dataset models/layer$L/2_align/protein_to_backbone.pickle    # that model's Even rows
  python reproduce/preprint.py --model-dir models/layer$L/4_fit --test-run tests/layer$L \
      --figures fig2a fig3a --out reproduce/layer$L
done
python reproduce/figS6.py --layer 1 reproduce/layer1 --layer 6 reproduce/layer6 \
    --layer 12 reproduce/v1_preprint --out reproduce/figS6.png