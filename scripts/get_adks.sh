#!/bin/bash

IND=$1
OUT=$(basename "$IND" "_protein.faa")
hmmsearch --tblout ${OUT}_PF00406_tblout.txt ~/db/PFam_2023/PF00406.hmm ${IND}_protein.faa