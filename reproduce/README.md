Refit model `models/refit`
```
roseate fitting-workflow --alignment data/high_qual_alignments/backbone.filtered_only.afa --train-ids data/predictive_models/supporting_data/even_sequence_ids.txt --traits data/high_qual_alignments/temp_data.tsv --scaler reference/v1_preprint/inputs/predictive_models/supporting_data/ogt_scaler.pickle -o models/refit --threads 8 --overwrite --verbose --set embed.contexts=models/v1_preprint/even_full_context.npy
```

Layer 1 model:

Layer 6 model:

