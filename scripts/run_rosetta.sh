#!/bin/bash

IND=$1

touch ${IND}_inputs.txt
echo "./test_structures2/${IND}_closed.pdb" >> ${IND}_inputs.txt

~/tools/rosetta.binary.linux.release-362/main/source/bin/rosetta_scripts.static.linuxgccrelease \
-l ${IND}_inputs.txt \
-parser:protocol ~/tools/OGT_project/rosetta/ogt_metrics.xml \
-out:file:score_only ./rosetta_out/${IND}_rosetta_out.sc

rm ${IND}_inputs.txt

