#!/bin/bash
# monitor_spades_pe.sh
# Usage: ./monitor_spades_pe.sh forward.fq reverse.fq output_dir

set -euo pipefail

FWD_IN=$1
REV_IN=$2
OUTDIR=$3

# Subsample sizes (number of reads per pair) and cores to test
READ_COUNTS=(10000 100000 1000000 10000000 15000000)   # adjust as needed
CORES=(15)                         # adjust as needed

# Optional SPAdes flag (set to "" if not needed)
ISOLATE_FLAG=""

# Sampling interval for RAM (seconds)
INTERVAL=5

mkdir -p "$OUTDIR/logs"
mkdir -p "$OUTDIR/results"

for reads in "${READ_COUNTS[@]}"; do
  # Subsample forward and reverse together
  FWD_SUB="$OUTDIR/sub_${reads}_1.fq"
  REV_SUB="$OUTDIR/sub_${reads}_2.fq"

  if [ ! -f "$FWD_SUB" ] || [ ! -f "$REV_SUB" ]; then
    echo "[INFO] Subsampling $reads read pairs..."
    seqtk sample -s100 "$FWD_IN" $reads > "$FWD_SUB"
    seqtk sample -s100 "$REV_IN" $reads > "$REV_SUB"
  fi

  for cores in "${CORES[@]}"; do
    echo "[INFO] Running SPAdes: ${reads} pairs, ${cores} cores"

    SAMPLE="sub${reads}_${cores}cores"
    RUN_OUT="$OUTDIR/results/${SAMPLE}_spades"
    LOGFILE="$OUTDIR/logs/mem_${SAMPLE}.csv"

    echo "running kmc"
    mkdir -p "$OUTDIR/tmp/"
    kmc -k55 "$FWD_SUB" "$OUTDIR/${SAMPLE}_kmc" "$OUTDIR/tmp/" > "$OUTDIR/${SAMPLE}_kmc_summary"

    # Start SPAdes with requested options
    spades.py -1 "$FWD_SUB" -2 "$REV_SUB" -o "$RUN_OUT" --meta $ISOLATE_FLAG -t "$cores" > /dev/null 2>&1 &
    pid=$!

    # find its process group id (PGID)
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    
    echo "timestamp,TotalRSS_MB" > "$LOGFILE"
    
    # monitor: sum RSS of ALL procs in that group
    while kill -0 "$pid" 2>/dev/null; do
      ts=$(date +%s)
      pids=$(pgrep -g "$pgid" | tr '\n' ' ')
      if [ -n "$pids" ]; then
        rss_mb=$(ps -o rss= -p $pids | awk '{s+=$1} END{printf "%.2f", s/1024}')  # KB→MB
        echo "$ts,$rss_mb" >> "$LOGFILE"
      fi
      sleep "$INTERVAL"
    done

    # optional peak summary
    peak=$(awk -F, 'NR>1 {if($2>m) m=$2} END{print m+0}' "$LOGFILE")
    echo "[SUMMARY] Peak total RSS = ${peak} MB for ${reads} pairs, ${cores} cores" | tee -a "$LOGFILE"


  done
done

echo "[DONE] All runs complete. Logs in $OUTDIR/logs/"
