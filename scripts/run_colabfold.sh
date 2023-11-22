#!/bin/bash

#while read IND; do 
IND=$1
echo "starting $IND"
colabfold_batch ./adk_sequences/${IND}.faa ./out_run2/${IND}_closed --templates --custom-template-path ./closed_templates/${IND} --overwrite-existing-results --amber --num-relax 3 --model-order 1,2 --random-seed 1040 --num-recycle 7 --save-recycles
colabfold_batch ./adk_sequences/${IND}.faa ./out_run2/${IND}_open --templates --custom-template-path ./open_templates/${IND} --overwrite-existing-results --amber --num-relax 3 --model-order 1,2 --random-seed 1040 --num-recycle 7 --save-recycles
# done  < $1
