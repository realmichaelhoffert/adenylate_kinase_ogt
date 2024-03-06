#!/bin/bash


IND=$1

# unpack and get top structure
tar -xzf ./outputs/${IND}_closed.tgz
pattern="${IND}_closed/*_relaxed_rank_001_*.pdb"
files=( $pattern )
top_struct=${files[0]}
echo "Top structure: $top_struct"

# copy top structure, save details to file
hash_og=$(sha256sum $top_struct | cut -f 1 -d" ")
echo -e "${top_struct}\t./test_structures/${IND}_closed.pdb\t${hash_og}" >> file_names.txt
cp $top_struct ./test_structures2/${IND}_closed.pdb

# remove file
rm -r "${IND}_closed"


