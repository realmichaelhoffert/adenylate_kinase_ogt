#!/bin/bash
while read acc; do
  [[ -f "./$acc.sra" ]] || ~/tools/sratoolkit.2.11.3-centos_linux64/bin/prefetch --output-directory . "$acc"
done < $1
