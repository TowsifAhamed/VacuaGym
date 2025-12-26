#!/bin/bash
# Quick test script to verify VacuaGym pipeline

# Use Python from virtual environment
PYTHON=.venv/bin/python

echo "=========================================="
echo "VacuaGym Quick Test"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

echo "Step 1: Testing data parsing..."
echo "  - Parsing CICY data..."
$PYTHON scripts/11_parse_cicy3.py > logs/test_parse_cicy.log 2>&1
if [ $? -eq 0 ]; then
    echo "    ✓ CICY parsing successful"
else
    echo "    ✗ CICY parsing failed (see logs/test_parse_cicy.log)"
fi

echo ""
echo "Step 2: Testing feature building..."
$PYTHON scripts/20_build_features.py > logs/test_features.log 2>&1
if [ $? -eq 0 ]; then
    echo "    ✓ Feature building successful"
else
    echo "    ✗ Feature building failed (see logs/test_features.log)"
fi

echo ""
echo "Step 3: Checking outputs..."

FILES=(
    "data/processed/tables/cicy3_configs.parquet"
    "data/processed/tables/cicy3_features.parquet"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "    ✓ $file ($size)"
    else
        echo "    ✗ $file (not found)"
    fi
done

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review logs in logs/ directory"
echo "  2. Run full pipeline: see TESTING_GUIDE.md"
echo "  3. Generate labels: .venv/bin/python scripts/30_generate_labels_toy_eft.py"
echo ""
