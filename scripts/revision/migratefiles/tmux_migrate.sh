#!/bin/bash

# Tmux Migration Session Script
# Creates a tmux session with multiple panes for monitoring

SESSION_NAME="file_migration"
SOURCE_DIR="/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding" #"${1:-}"
DEST_DIR="/vast/labs/DBIC/datasets/Life/life-encoding" #"${2:-}"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed"
    echo "Install with: sudo apt-get install tmux  (Debian/Ubuntu)"
    echo "           or: sudo yum install tmux      (RHEL/CentOS)"
    exit 1
fi

# Validate arguments
if [[ -z "$SOURCE_DIR" ]] || [[ -z "$DEST_DIR" ]]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    echo ""
    echo "Example: $0 /path/to/source /path/to/destination"
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new tmux session
echo "Creating tmux session: $SESSION_NAME"

# Create session with first window
tmux new-session -d -s "$SESSION_NAME" -n "Migration"

# Split window into panes
# Layout: 
#   ┌─────────────┬─────────────┐
#   │             │             │
#   │   Main      │   Monitor   │
#   │ Migration   │   Dest Dir  │
#   │             │             │
#   ├─────────────┴─────────────┤
#   │        Log Viewer         │
#   └───────────────────────────┘

# Main pane (top-left): Run migration script
# tmux send-keys -t "$SESSION_NAME:0.0" "cd /home/claude" C-m
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmux send-keys -t "$SESSION_NAME:0.0" "cd '$SCRIPT_DIR'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "chmod +x migrate_files.sh" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo 'Ready to start migration'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo 'Press Enter to begin...'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo ''" C-m

# Split vertically (create top-right pane)
tmux split-window -h -t "$SESSION_NAME:0"
tmux send-keys -t "$SESSION_NAME:0.1" "watch -n 2 \"echo 'Destination Directory: $DEST_DIR' && echo '' && du -sh '$DEST_DIR' 2>/dev/null && echo '' && find '$DEST_DIR' -type f 2>/dev/null | wc -l | xargs echo 'Files:' && echo '' && find '$SOURCE_DIR' -type f 2>/dev/null | wc -l | xargs echo 'Source files remaining:'\"" C-m

# Split horizontally (create bottom pane across full width)
tmux split-window -v -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.2" "echo 'Waiting for migration to start...'" C-m
tmux send-keys -t "$SESSION_NAME:0.2" "echo 'Log file will appear here once migration begins.'" C-m

# Resize panes - make bottom pane smaller
tmux resize-pane -t "$SESSION_NAME:0.2" -y 10

# Select the main pane
tmux select-pane -t "$SESSION_NAME:0.0"

# Create a command to run in the main pane (but don't execute yet)
tmux send-keys -t "$SESSION_NAME:0.0" "./migrate_files.sh '$SOURCE_DIR' '$DEST_DIR'"

# Send command to bottom pane to tail the log once it exists
LOGFILE="migration_\$(date +%Y%m%d)*.log"
tmux send-keys -t "$SESSION_NAME:0.2" "sleep 5 && tail -f migration_*.log 2>/dev/null || echo 'Waiting for log file...'" C-m

# Create second window for checksums
tmux new-window -t "$SESSION_NAME:1" -n "Checksums"
tmux send-keys -t "$SESSION_NAME:1" "echo 'Checksum verification window'" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo 'Commands you can run here:'" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo '  - cat checksums_*.txt | head -20    # View checksums'" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo '  - sha256sum -c checksums_*.txt      # Verify all checksums'" C-m
tmux send-keys -t "$SESSION_NAME:1" "echo ''" C-m

# Create third window for manual intervention if needed
tmux new-window -t "$SESSION_NAME:2" -n "Shell"
# tmux send-keys -t "$SESSION_NAME:2" "cd /home/claude" C-m
tmux send-keys -t "$SESSION_NAME:2" "cd '$SCRIPT_DIR'" C-m
tmux send-keys -t "$SESSION_NAME:2" "echo 'Emergency shell - use for manual operations if needed'" C-m

# Select the migration window
tmux select-window -t "$SESSION_NAME:0"

# Attach to the session
echo ""
echo "Tmux session created successfully!"
echo ""
echo "Session layout:"
echo "  Window 0 (Migration): Main migration + monitoring + logs"
echo "  Window 1 (Checksums): Checksum verification tools"
echo "  Window 2 (Shell):     Emergency shell"
echo ""
echo "Tmux commands:"
echo "  Ctrl+b, n        - Next window"
echo "  Ctrl+b, p        - Previous window"
echo "  Ctrl+b, 0/1/2    - Go to window 0/1/2"
echo "  Ctrl+b, arrow    - Navigate between panes"
echo "  Ctrl+b, d        - Detach from session"
echo "  Ctrl+b, [        - Scroll mode (q to exit)"
echo ""
echo "To reattach later: tmux attach -t $SESSION_NAME"
echo ""
echo "Attaching to session..."
sleep 2

tmux attach -t "$SESSION_NAME"