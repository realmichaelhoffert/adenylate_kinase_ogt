## v1_preprint:
`cp reference/v1_preprint/outputs/predictive_models/even_enetcv_ind.pickle models/v1_preprint/model.pickle`
`cp reference/v1_preprint/outputs/predictive_models/ogt_scaler.pickle models/v1_preprint/scaler.pickle`
`cp reference/v1_preprint/inputs/predictive_models/supporting_data/even_msa_dataset.pickle models/v1_preprint/context_dataset.pickle`  

Baseline: `python reproduce/preprint.py --reference reference/v1_preprint/ --model-dir models/v1_preprint/ --out reproduce/v1_preprint --sites legacy_fig2a`

## Refit model `models/refit`
```
roseate fitting-workflow --alignment data/high_qual_alignments/backbone.filtered_only.afa --train-ids data/predictive_models/supporting_data/even_sequence_ids.txt --traits data/high_qual_alignments/temp_data.tsv --scaler reference/v1_preprint/inputs/predictive_models/supporting_data/ogt_scaler.pickle -o models/refit --threads 8 --overwrite --verbose --set embed.contexts=models/v1_preprint/even_full_context.npy
```
## Check refit matches archived
```
python -c "from roseate import evaluate, io; \
d = evaluate.compare_models(io.reload_pickle('models/refit/4_fit/model.pickle'), io.reload_pickle('models/v1_preprint/model.pickle'), atol=1e-5); \
print('\n'.join(d[:20]) or 'models match')"
> site 151 alpha_: max abs diff 0.0166 > 1e-05
> site 151 coef_: max abs diff 0.00386 > 1e-05
> full_predictions: shape (215, 0) != (215, 230)
> predictions: shape (0,) != (230,)
> true_values: shape (0,) != (230,)
> true_labels differ
```


## Layer 1 model
```
roseate fitting-workflow --alignment data/high_qual_alignments/backbone.filtered_only.afa --train-ids data/predictive_models/supporting_data/even_sequence_ids.txt --traits data/high_qual_alignments/temp_data.tsv --scaler reference/v1_preprint/inputs/predictive_models/supporting_data/ogt_scaler.pickle -o models/layer1 --threads 8 --overwrite --verbose --set embed.contexts=models/v1_preprint/even_full_context.npy --set embed.layer=1
```

## Layer 6 model:
```
roseate fitting-workflow --alignment data/high_qual_alignments/backbone.filtered_only.afa --train-ids data/predictive_models/supporting_data/even_sequence_ids.txt --traits data/high_qual_alignments/temp_data.tsv --scaler reference/v1_preprint/inputs/predictive_models/supporting_data/ogt_scaler.pickle -o models/layer6 --threads 8 --overwrite --verbose --set embed.contexts=models/v1_preprint/even_full_context.npy --set embed.layer=6
```
## Arthrobacter
`roseate prediction-workflow --genomes ./data/validation_datasets/Arthrobacter/genomes/*.fasta -o examples/Arthrobacter --threads 8 --verbose`
## Mycobacteria
`roseate prediction-workflow --genomes ./data/validation_datasets/Mycobacteria/genomes/*.fasta -o examples/Mycobacteria --threads 8 --verbose`

Differences in predictions are reasonable and will be clarified in further testing of embedding properties

## Fig S6 reproduction

`bash reproduce/figS6.sh`

## Metagenomic workflow