while read accession; do
  ~/tools/sratoolkit.2.11.3-centos_linux64/bin/fasterq-dump.2.11.3 "$accession" -O ./ -S
done < $1
