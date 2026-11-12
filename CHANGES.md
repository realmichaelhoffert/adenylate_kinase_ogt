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
