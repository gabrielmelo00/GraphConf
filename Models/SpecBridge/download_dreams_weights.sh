#!/bin/bash
# Download DreaMS pre-trained weights from Zenodo
# Downloads DreaMS base model weights to runs/DreaMS/

set -e

# Configuration
ZENODO_RECORD="10997887"
ZENODO_URL="https://zenodo.org/api/records/${ZENODO_RECORD}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================"
echo "DreaMS Pre-trained Weights Download Script"
echo "============================================================"
echo ""

# File mapping: "zenodo_filename|local_path"
declare -A FILE_MAP=(
    # DreaMS base model checkpoint
    ["ssl_model.ckpt"]="runs/DreaMS/ssl_model.ckpt"
)

# Get file list from Zenodo
echo "📦 Fetching file list from Zenodo..."
FILES_JSON=$(curl -s "${ZENODO_URL}")

if [ -z "$FILES_JSON" ]; then
    echo "❌ Failed to fetch Zenodo record. Check your internet connection."
    exit 1
fi

# Extract file URLs and names
echo "✓ Connected to Zenodo record: ${ZENODO_RECORD}"
echo ""

# Download each file
for zenodo_name in "${!FILE_MAP[@]}"; do
    local_path="${FILE_MAP[$zenodo_name]}"
    
    # Extract download URL for this file
    download_url=$(echo "$FILES_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data.get('files', []):
    if f.get('key') == '${zenodo_name}':
        print(f['links']['self'])
        break
" 2>/dev/null)
    
    if [ -z "$download_url" ]; then
        echo -e "${YELLOW}⚠️  File not found on Zenodo: ${zenodo_name}${NC}"
        continue
    fi
    
    # Create directory if needed
    local_dir=$(dirname "$local_path")
    if [ ! -d "$local_dir" ]; then
        echo "📁 Creating directory: $local_dir"
        mkdir -p "$local_dir"
    fi
    
    # Check if file already exists
    if [ -f "$local_path" ]; then
        echo -e "${BLUE}⊘ Skipping: ${zenodo_name} (already exists at ${local_path})${NC}"
        echo ""
        continue
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}📥 Downloading: ${zenodo_name}${NC}"
    echo "   → ${local_path}"
    
    # Get file size for progress
    file_size=$(echo "$FILES_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data.get('files', []):
    if f.get('key') == '${zenodo_name}':
        print(f.get('size', 0))
        break
" 2>/dev/null)
    
    if [ -n "$file_size" ] && [ "$file_size" != "0" ]; then
        size_mb=$((file_size / 1024 / 1024))
        echo "   Size: ${size_mb} MB"
    fi
    echo ""
    
    # Download file
    if curl -L -o "$local_path" --progress-bar "$download_url"; then
        echo -e "${GREEN}✓ Downloaded: ${zenodo_name} → ${local_path}${NC}"
    else
        echo -e "${YELLOW}❌ Failed to download: ${zenodo_name}${NC}"
        # Remove partial file
        [ -f "$local_path" ] && rm "$local_path"
    fi
    echo ""
done

echo "============================================================"
echo -e "${GREEN}✅ Download complete!${NC}"
echo ""
echo "DreaMS pre-trained weights downloaded to:"
echo "  • runs/DreaMS/ssl_model.ckpt"
echo ""
echo "Usage in training/evaluation:"
echo "  --dreams-ckpt runs/DreaMS/ssl_model.ckpt"
echo "============================================================"
