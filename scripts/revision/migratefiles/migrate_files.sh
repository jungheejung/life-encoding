#!/bin/bash

# Safe File Migration Script with Checksum Verification
# This script handles partially moved files and verifies integrity

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SOURCE_DIR="${1:-}"
DEST_DIR="${2:-}"
LOG_FILE="migration_$(date +%Y%m%d_%H%M%S).log"
CHECKSUM_FILE="checksums_$(date +%Y%m%d_%H%M%S).txt"

# Function to print colored output
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to calculate checksum
calculate_checksum() {
    local file="$1"
    sha256sum "$file" | awk '{print $1}'
}

# Function to check if file exists in destination
check_destination() {
    local rel_path="$1"
    local dest_file="${DEST_DIR}/${rel_path}"
    
    if [[ -f "$dest_file" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to verify file integrity
verify_file() {
    local source_file="$1"
    local dest_file="$2"
    
    local source_checksum=$(calculate_checksum "$source_file")
    local dest_checksum=$(calculate_checksum "$dest_file")
    
    if [[ "$source_checksum" == "$dest_checksum" ]]; then
        return 0
    else
        log_error "Checksum mismatch: $source_file"
        log_error "  Source: $source_checksum"
        log_error "  Dest:   $dest_checksum"
        return 1
    fi
}

# Function to safely move a file
safe_move_file() {
    local source_file="$1"
    local rel_path="$2"
    local dest_file="${DEST_DIR}/${rel_path}"
    local dest_dir=$(dirname "$dest_file")
    
    # Create destination directory if it doesn't exist
    mkdir -p "$dest_dir"
    
    # Calculate source checksum before moving
    local source_checksum=$(calculate_checksum "$source_file")
    echo "${source_checksum}  ${rel_path}" >> "$CHECKSUM_FILE"
    
    # Copy file to destination
    cp -p "$source_file" "$dest_file"
    
    # Verify the copy
    if verify_file "$source_file" "$dest_file"; then
        log_success "Verified: $rel_path"
        # Remove source file only after verification
        rm "$source_file"
        log_success "Moved: $rel_path"
        return 0
    else
        log_error "Verification failed: $rel_path - keeping source file"
        rm "$dest_file"
        return 1
    fi
}

# Function to handle already moved files
handle_existing_file() {
    local source_file="$1"
    local dest_file="$2"
    local rel_path="$3"
    
    log_warning "File already exists in destination: $rel_path"
    
    # Check if they're identical
    if verify_file "$source_file" "$dest_file"; then
        log_success "Files are identical, safe to remove source: $rel_path"
        rm "$source_file"
        echo "$(calculate_checksum "$dest_file")  ${rel_path}" >> "$CHECKSUM_FILE"
        return 0
    else
        log_error "Files differ! Manual intervention required: $rel_path"
        log_error "  Source: $source_file"
        log_error "  Dest:   $dest_file"
        return 1
    fi
}

# Main migration function
migrate_files() {
    local moved_count=0
    local already_moved_count=0
    local error_count=0
    local total_count=0
    
    log_info "Starting migration from $SOURCE_DIR to $DEST_DIR"
    log_info "Log file: $LOG_FILE"
    log_info "Checksum file: $CHECKSUM_FILE"
    echo ""
    
    # Find all files in source directory
    while IFS= read -r -d '' source_file; do
        ((total_count++))
        
        # Get relative path
        rel_path="${source_file#$SOURCE_DIR/}"
        dest_file="${DEST_DIR}/${rel_path}"
        
        log_info "Processing ($total_count): $rel_path"
        
        # Check if file already exists in destination
        if check_destination "$rel_path"; then
            if handle_existing_file "$source_file" "$dest_file" "$rel_path"; then
                ((already_moved_count++))
            else
                ((error_count++))
            fi
        else
            if safe_move_file "$source_file" "$rel_path"; then
                ((moved_count++))
            else
                ((error_count++))
            fi
        fi
        
        echo ""
    done < <(find "$SOURCE_DIR" -type f -print0)
    
    # Summary
    echo ""
    log_info "========== Migration Summary =========="
    log_info "Total files processed: $total_count"
    log_success "Newly moved files: $moved_count"
    log_warning "Already moved files: $already_moved_count"
    log_error "Errors: $error_count"
    log_info "======================================"
    
    # Check if source directory is now empty
    if [[ -d "$SOURCE_DIR" ]] && [[ -z "$(find "$SOURCE_DIR" -type f)" ]]; then
        log_success "Source directory is now empty of files"
        log_info "You can safely remove empty directories with: find '$SOURCE_DIR' -type d -empty -delete"
    fi
}

# Validate arguments
if [[ -z "$SOURCE_DIR" ]] || [[ -z "$DEST_DIR" ]]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    echo ""
    echo "Example: $0 /path/to/source /path/to/destination"
    exit 1
fi

# Validate source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    log_error "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Ask for confirmation
echo -e "${YELLOW}WARNING:${NC} This will migrate files from:"
echo -e "  Source:      ${BLUE}$SOURCE_DIR${NC}"
echo -e "  Destination: ${BLUE}$DEST_DIR${NC}"
echo ""
read -p "Continue? (yes/no): " -r
echo ""

if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    log_info "Migration cancelled by user"
    exit 0
fi

# Run migration
migrate_files

log_info "Migration complete!"