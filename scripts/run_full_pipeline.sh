#!/bin/bash
# Run full dataset label generation with resource limits
# This prevents the system from freezing

set -e

echo "======================================================================="
echo "VacuaGym: Full Dataset Label Generation (Resource-Limited)"
echo "======================================================================="
echo ""
echo "Resource limits:"
echo "  • Lower CPU priority (nice +10)"
echo "  • Lower I/O priority (ionice -c2 -n7)"
echo "  • Memory-optimized batching"
echo "  • Garbage collection every 500 samples"
echo ""
echo "Estimated time: ~2-3 hours (with resource limits)"
echo "Checkpoint location: data/processed/labels/checkpoints/"
echo ""

# Check if running in tmux
if [ -z "$TMUX" ]; then
    echo "⚠️  WARNING: Not running in tmux"
    echo "   If SSH disconnects, the process will stop"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Tip: Start with: tmux new -s vacuagym"
        exit 1
    fi
fi

echo "Starting at $(date)"
echo ""

# Run with lower priority to avoid system freeze
# nice -n 10: Lower CPU priority
# ionice -c2 -n7: Lower I/O priority (best-effort class, low priority)
nice -n 10 ionice -c2 -n7 .venv/bin/python scripts/30_generate_labels_toy_eft.py

echo ""
echo "======================================================================="
echo "Complete at $(date)"
echo "======================================================================="
