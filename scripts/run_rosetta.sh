#!/bin/bash

# run:
# parallel -j 12 './run_rosetta.sh -i {} -x ./path/to/your_metrics.xml' :::: inds_rosetta.txt

# --- Defaults ---
ROSETTA_BIN="$HOME/tools/rosetta.binary.linux.release-362/main/source/bin/rosetta_scripts.static.linuxgccrelease"
STRUCT_DIR="./test_structures"
OUT_DIR="./rosetta_out"
XML_FILE=""
IND=""

# --- Help Documentation ---
usage() {
    echo "Usage: $0 -i <ID> -x <xml_path> [-d <struct_dir>] [-b <binary_path>] [-o <out_dir>]"
    echo ""
    echo "Required:"
    echo "  -i    The Index/ID of the structure (e.g., from inds_rosetta.txt)"
    echo "  -x    Path to the Rosetta XML protocol file"
    echo ""
    echo "Optional:"
    echo "  -d    Directory where PDBs live (Default: $STRUCT_DIR)"
    echo "  -b    Path to rosetta_scripts binary (Default: $ROSETTA_BIN)"
    echo "  -o    Output directory for score files (Default: $OUT_DIR)"
    echo "  -h    Display this help message"
    exit 1
}

# --- Parse Arguments ---
while getopts "i:x:d:b:o:h" opt; do
    case ${opt} in
        i) IND=$OPTARG ;;
        x) XML_FILE=$OPTARG ;;
        d) STRUCT_DIR=$OPTARG ;;
        b) ROSETTA_BIN=$OPTARG ;;
        o) OUT_DIR=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# --- Validation ---
if [[ -z "$IND" || -z "$XML_FILE" ]]; then
    echo "Error: ID (-i) and XML file (-x) are required."
    usage
fi

if [[ ! -f "$XML_FILE" ]]; then
    echo "Error: XML file not found at $XML_FILE"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUT_DIR"

# --- Rosetta Execution ---
# Create a localized input list for this specific ID
INPUT_LIST="${IND}_inputs.txt"
echo "${STRUCT_DIR}/${IND}.pdb" > "$INPUT_LIST"

echo "Processing ID: $IND using $XML_FILE"
ROSETTA_LOG=${OUT_DIR}/${IND}_rosetta.log
echo "Rosetta log: $ROSETTA_LOG"

"$ROSETTA_BIN" \
    -out:level 300 \
    -l "$INPUT_LIST" \
    -parser:protocol "$XML_FILE" \
    -out:file:score_only "${OUT_DIR}/${IND}_rosetta_out.sc" \
    -overwrite > $ROSETTA_LOG # Optional: ensures Rosetta doesn't skip if file exists

# --- Cleanup ---
rm "$INPUT_LIST"
