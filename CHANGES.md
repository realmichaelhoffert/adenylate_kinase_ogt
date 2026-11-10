## November 2026:
Completed refactoring of code into `roseate/` package to improve usability.

## Changes with effects on model:
* Scalers now fit with scaler trained on actual alignment, not incorrect stored legacy scaler    
* Changes to site aggregation procedure (ensembling predictions across sites) to avoid transductive site selection  
* Precompute = False in ElasticNetCV to avoid float64 error  
* Phylogenetic CV: glmnet-based ElasticNet swapped for seeded sklearn ElasticNet and ElasticNetCV  

## Chages with no effects
* per_site_predict: fixed potential bug in order in which sites are retrieved from model  
* Don't store trait information in fasta sequence headers, instead require file  
* Remove XXXtemp=XXX field from fasta headers
* Fix script bug in run_esm_pipeline.sh
* io.save_split function to save training / test splits
* nn_models.train_model sizes neural network from inputs rather than hard-coded `768`
