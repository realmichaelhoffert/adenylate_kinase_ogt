## November 2026:
Completed refactoring of code into `roseate/` package to improve usability.

## Changes with effects on model:
* Scalers now fit with scaler trained on actual alignment, not incorrect stored legacy scaler: RMSEs × 0.9966, with R² and Spearman unchanged.
* Changes to site aggregation procedure (ensembling predictions across sites) to avoid transductive site selection, using all sites by default (specified with all), but can be replaced with training-only CV selection and legacy_fig2a selection.
* Precompute = False in ElasticNetCV to avoid float64 error: scikit-learn’s float32 Gram-matrix check causes refitting of alpha_ values to now differ from the archived model at around 1e-9.
* Phylogenetic CV: glmnet-based ElasticNet swapped for seeded sklearn ElasticNet and ElasticNetCV  

## Changes with no effects
* per_site_predict: fixed potential bug in order in which sites are retrieved from model  
* Don't store trait information in fasta sequence headers, instead require file  
* Remove XXXtemp=XXX field from fasta headers
* Fix script bug in run_esm_pipeline.sh
* io.save_split function to save training / test splits
* nn_models.train_model sizes neural network from inputs rather than hard-coded `768`

## Changes on dev_workflow:
* Add mode switch to script which runs prodigal. `extract --mode` selects mode.
* add --genome-proteins and --metagenome-proteins which recycle workflow from --genomes and --metagenomes, just skipping annotation step
* add --sequences option which skips annotation workflow: add protein check to --sequences to avoid users putting in garbo sequences
* in `cmd_extract`, write a `1_extract/genomes` folder with symlinks to input genomes or metagenomes for annotation
* add `align --train-ids`, which allows only specific sequences to be extracted from the alignment for training (e.g. the even dataset)
* added `prediction-workflow`
    * reuses input parsers for each sub-command and a pre-written config referencing the preprint model (will be changed to "current" model)
* added `fitting-workflow`
    * same design as prediction workflow, but starts with align.
    * a fitting workflow writes `model_config.yaml` within the output folder, so that a NEW prediction workflow can pass --config `fitted_model/model_config.yaml` using the newly fitted model.
