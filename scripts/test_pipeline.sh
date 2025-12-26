#!/bin/bash
# Test script to verify the improved pipeline produces multi-class labels

set -e

PYTHON=.venv/bin/python

echo "======================================================================"
echo "VacuaGym Pipeline Test - Multi-Class Label Generation"
echo "======================================================================"
echo ""

# Step 1: Reparse CICY with proper feature extraction
echo "Step 1: Reparsing CICY data with full feature extraction..."
$PYTHON scripts/11_parse_cicy3.py

# Step 2: Rebuild features
echo ""
echo "Step 2: Rebuilding features..."
$PYTHON scripts/20_build_features.py

# Step 3: Generate labels with new non-convex potential
echo ""
echo "Step 3: Generating labels (N_LIMIT=100 for quick test)..."
echo "   This will use the new physics-inspired non-convex potential"
echo "   Expected: Multiple stability classes (stable, saddle, failed, marginal)"
$PYTHON scripts/30_generate_labels_toy_eft.py

# Step 4: Verify label diversity
echo ""
echo "Step 4: Verifying label diversity..."
$PYTHON -c "
import pandas as pd

labels = pd.read_parquet('data/processed/labels/toy_eft_stability.parquet')

print('=' * 70)
print('LABEL VERIFICATION RESULTS')
print('=' * 70)
print()
print(f'Total labeled samples: {len(labels):,}')
print()
print('Stability distribution:')
print(labels['stability'].value_counts())
print()
print(f'Number of classes: {labels[\"stability\"].nunique()}')
print()

if labels['stability'].nunique() == 1:
    print('❌ FAILED: Still only one class!')
    print('   The potential may need further tuning.')
    exit(1)
else:
    print(f'✅ SUCCESS: Found {labels[\"stability\"].nunique()} different classes!')
    print()
    print('Validation metrics:')
    print(f'  Minimization success rate: {labels[\"minimization_success\"].mean() * 100:.1f}%')
    print(f'  Gradient norm (mean): {labels[labels[\"minimization_success\"]][\"grad_norm\"].mean():.2e}')
    print(f'  Condition number (mean): {labels[labels[\"minimization_success\"]][\"condition_number\"].mean():.2e}')
    print()
    print('By dataset:')
    print(labels.groupby(['dataset', 'stability']).size().unstack(fill_value=0))
"

echo ""
echo "======================================================================"
echo "Pipeline test complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. If multi-class labels verified, create splits and train models"
echo "  2. Run: .venv/bin/python scripts/40_make_splits.py"
echo "  3. Run: .venv/bin/python scripts/50_train_baseline_tabular.py"
echo "  4. Open VacuaGym_Pipeline.ipynb and run full pipeline"
echo ""
