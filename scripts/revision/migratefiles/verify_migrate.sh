#!/bin/bash

# Post-Migration Verification Script
# Verifies all files were correctly migrated using checksums

set -euo pipefail

DEST_DIR="${1:-}"
CHECKSUM_FILE="${2:-}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [[ -z "$DEST_DIR" ]] || [[ -z "$CHECKSUM_FILE" ]]; then
    echo "Usage: $0 <destination_directory> <checksum_file>"
    echo ""
    echo "Example: $0 /path/to/destination checksums_20250110_123456.txt"
    exit 1
fi

if [[ ! -f "$CHECKSUM_FILE" ]]; then
    echo -e "${RED}Error:${NC} Checksum file not found: $CHECKSUM_FILE"
    exit 1
fi

if [[ ! -d "$DEST_DIR" ]]; then
    echo -e "${RED}Error:${NC} Destination directory not found: $DEST_DIR"
    exit 1
fi

echo "Verifying migration to: $DEST_DIR"
echo "Using checksum file: $CHECKSUM_FILE"
echo ""

total=0
verified=0
failed=0
missing=0

while IFS= read -r line; do
    ((total++))
    
    # Parse checksum and filename
    checksum=$(echo "$line" | awk '{print $1}')
    filename=$(echo "$line" | cut -d' ' -f3-)
    filepath="${DEST_DIR}/${filename}"
    
    # Check if file exists
    if [[ ! -f "$filepath" ]]; then
        echo -e "${RED}MISSING:${NC} $filename"
        ((missing++))
        continue
    fi
    
    # Calculate current checksum
    current_checksum=$(sha256sum "$filepath" | awk '{print $1}')
    
    # Compare checksums
    if [[ "$checksum" == "$current_checksum" ]]; then
        echo -e "${GREEN}OK:${NC} $filename"
        ((verified++))
    else
        echo -e "${RED}FAILED:${NC} $filename"
        echo "  Expected: $checksum"
        echo "  Got:      $current_checksum"
        ((failed++))
    fi
done < "$CHECKSUM_FILE"

echo ""
echo "========== Verification Summary =========="
echo "Total files:     $total"
echo -e "${GREEN}Verified:        $verified${NC}"
echo -e "${RED}Failed:          $failed${NC}"
echo -e "${YELLOW}Missing:         $missing${NC}"
echo "=========================================="

if [[ $failed -eq 0 ]] && [[ $missing -eq 0 ]]; then
    echo -e "${GREEN}✓ All files verified successfully!${NC}"
    exit 0
else
    echo -e "${RED}✗ Verification completed with errors${NC}"
    exit 1
fi