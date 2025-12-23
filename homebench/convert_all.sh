#!/bin/bash
# Convert all HomeBench datasets from original to ThingDescription format

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/../datasets/HomeBench/original"
OUTPUT_DIR="${SCRIPT_DIR}/../datasets/HomeBench/converted"
CONVERTER="${SCRIPT_DIR}/ground_truth_converter.py"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Check if rdflib is installed
if ! python3 -c "import rdflib" 2>/dev/null; then
    echo "Error: rdflib is not installed"
    echo "Please install it with: pip install rdflib"
    exit 1
fi

echo "Converting HomeBench datasets..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Convert each JSONL file
for input_file in "$INPUT_DIR"/*.jsonl; do
    if [ -f "$input_file" ]; then
        filename=$(basename "$input_file" .jsonl)
        output_file="$OUTPUT_DIR/${filename}.json"

        echo "Converting: $filename.jsonl -> $filename.json"
        python3 "$CONVERTER" -i "$input_file" -o "$output_file"
        echo ""
    fi
done

echo "All conversions complete!"
echo "Converted files are in: $OUTPUT_DIR"
