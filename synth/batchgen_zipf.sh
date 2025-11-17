#!/bin/bash

set -e
if [ $# -ne 1 ]; then
    echo "Usage: bash batchgen_zipf.sh <count_of_elements>"
    exit 1
fi
nr_elem=$1
# Change to script directory
cd "$(dirname "$0")"
mkdir -p zfcols/{04,08,12,16,20}_{16,32}

for skew in 04 08 12 16 20; do
  max_dimval=$((nr_elem / 100))

  o="zfcols/${skew}_16"
  # Generate attribute columns
  python3 onegen_zipf.py "1.${skew}" 65536 "$nr_elem" "$o/a1" &
  python3 onegen_zipf.py "1.${skew}" 65536 "$nr_elem" "$o/a2" &
  # Generate foreign keys
  python3 onegen_zipf.py "1.${skew}" "$max_dimval" "$nr_elem" "$o/fk" &

  o="zfcols/${skew}_32"
  # Generate attribute columns
  python3 onegen_zipf.py "1.${skew}" 4294967296 "$nr_elem" "$o/a1" &
  python3 onegen_zipf.py "1.${skew}" 4294967296 "$nr_elem" "$o/a2" &
  # Generate foreign keys
  python3 onegen_zipf.py "1.${skew}" "$max_dimval" "$nr_elem" "$o/fk" &
done

wait
echo "Batch generation complete!"
