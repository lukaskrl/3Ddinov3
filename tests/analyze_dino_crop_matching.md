# DINO Loss and Crop Matching Analysis for CT Scans

## Problem Statement

The model is experiencing **uniform collapse** - outputs are becoming uniform across all prototypes. This is likely because:

1. **CT scans have lots of background** (air, empty space) - often 50-80% of the volume
2. **Local crops are matched to global crops** - if both contain mostly background, the model learns "background → background"
3. **The model should learn fine-grained details within the human body**, not just "body vs background"
4. **Foreground-biased cropping is currently disabled** (`foreground_crop_prob: 0`)

## Current DINO Loss Matching Strategy

### How DINO Loss Works

From `dinov3/train/ssl_meta_arch.py:847-869`:

1. **DINO Global Loss:**
   - Matches: `student(global_crops) ↔ teacher(global_crops)`
   - Purpose: Learn that different views/augmentations of the same volume should have similar representations
   - Example: Two different global crops from the same CT scan should match

2. **DINO Local Loss:**
   - Matches: `student(local_crops) ↔ teacher(global_crops)`
   - Purpose: Learn that local details should match the global context
   - **PROBLEM**: If local crop is mostly background and global crop is mostly background, model learns "background matches background"

### Current Crop Generation

From `dinov3/data/augmentations_3d.py`:

1. **Global Crops:**
   - `use_foreground_bias=True` BUT `foreground_crop_prob=0` → **Not actually used!**
   - Randomly sampled from anywhere in the volume
   - Can be mostly background

2. **Local Crops:**
   - `use_foreground_bias=True` BUT `foreground_crop_prob=0` → **Not actually used!**
   - `local_crops_subset_of_global_crops=False` → Independent sampling
   - Randomly sampled from anywhere in the volume
   - Can be mostly background, completely unrelated to global crops

3. **Config Settings:**
   ```yaml
   foreground_threshold: 500    # HU threshold (above = foreground)
   foreground_crop_prob: 0       # DISABLED! Should be > 0
   min_foreground_ratio: 0.3     # Set but not used (because prob=0)
   localcrops_subset_of_globalcrops: false  # Local crops independent
   ```

## Root Causes

### 1. Background Matching Problem
- **Scenario**: Local crop = 80% background, Global crop = 70% background
- **What model learns**: "Background regions match background regions"
- **What we want**: "Fine-grained anatomical details match their context"

### 2. Independent Crop Sampling
- Local crops are sampled independently from global crops
- No guarantee they share context or even overlap
- Model can't learn "local detail within global context"

### 3. No Foreground Guarantee
- Even with `min_foreground_ratio: 0.3`, it's not enforced because `foreground_crop_prob: 0`
- Crops can be 100% background
- Model wastes capacity learning background patterns

### 4. Loss Doesn't Weight by Foreground Content
- All crops contribute equally to loss
- Background-heavy crops get same weight as foreground-rich crops
- Model optimizes for matching background (easier, more common)

## Proposed Solutions

### Solution 1: Enable Foreground-Biased Cropping (Quick Fix)

**Changes:**
```yaml
foreground_crop_prob: 1.0  # Always use foreground-biased cropping
min_foreground_ratio: 0.5  # Require at least 50% foreground in crops
```

**Pros:**
- Simple, immediate fix
- Ensures crops contain meaningful content
- No code changes needed

**Cons:**
- Still matches local to global independently
- Doesn't guarantee local crops are within global crops
- May fail if volume has very little foreground

### Solution 2: Make Local Crops Subsets of Global Crops

**Changes:**
```yaml
localcrops_subset_of_globalcrops: true  # Local crops inside global crops
```

**How it works:**
- Local crops are sampled from within the global crop volume
- Ensures local crops share context with global crops
- Model learns "local detail within this specific global context"

**Pros:**
- Guarantees spatial relationship
- More interpretable learning signal
- Matches DINOv2's approach for images

**Cons:**
- Local crops constrained to global crop region
- Less diversity in local crop locations
- If global crop is mostly background, local crops will be too

### Solution 3: Foreground-Aware Loss Weighting

**New Feature:**
- Compute foreground ratio for each crop
- Weight loss contribution by foreground ratio
- Background-heavy crops contribute less to loss

**Implementation:**
```python
# In compute_losses()
foreground_ratios_global = compute_foreground_ratio(global_crops)
foreground_ratios_local = compute_foreground_ratio(local_crops)

# Weight losses by foreground content
dino_global_loss_weighted = dino_global_loss * foreground_ratios_global.mean()
dino_local_loss_weighted = dino_local_loss * foreground_ratios_local.mean()
```

**Pros:**
- Prioritizes learning from meaningful content
- Still allows some background matching (but weighted less)
- Flexible - can tune weighting function

**Cons:**
- Requires computing foreground masks during training (overhead)
- Need to decide weighting function (linear? squared? threshold?)

### Solution 4: Hierarchical Matching Strategy

**New Approach:**
Instead of matching local→global, use hierarchical matching:

1. **Global-to-Global**: Same as current (different views match)
2. **Local-to-Local**: Match local crops to each other (fine-grained details)
3. **Local-to-Global**: Only if local crop is within foreground region of global crop

**Implementation:**
```python
# Only match local to global if local has sufficient foreground
if local_foreground_ratio > threshold:
    dino_local_to_global_loss = compute_loss(student_local, teacher_global)
else:
    # Match local to local instead
    dino_local_to_local_loss = compute_loss(student_local, teacher_local)
```

**Pros:**
- More sophisticated matching strategy
- Adapts to crop content
- Can learn both fine-grained (local-local) and contextual (local-global)

**Cons:**
- More complex implementation
- Need to handle different loss combinations
- May need to tune thresholds

### Solution 5: Foreground-Only Matching

**Approach:**
- Only compute loss for crops that meet foreground threshold
- Skip background-heavy crops entirely
- Or use a separate "background" prototype for background regions

**Pros:**
- Forces model to focus on meaningful content
- Clear separation of foreground vs background learning

**Cons:**
- May waste crops (if many are background)
- Need to handle variable batch sizes
- Background still needs some representation

### Solution 6: Multi-Scale Foreground Cropping

**Approach:**
- Ensure global crops have high foreground ratio (e.g., 70%)
- Ensure local crops have high foreground ratio (e.g., 60%)
- Make local crops subsets of global crops
- Use foreground-biased sampling for both

**Combination:**
```yaml
foreground_crop_prob: 1.0
min_foreground_ratio: 0.6  # Higher threshold
localcrops_subset_of_globalcrops: true
```

**Pros:**
- Combines multiple strategies
- Strong guarantee of meaningful content
- Maintains spatial relationships

**Cons:**
- May be too restrictive
- Could fail on volumes with sparse foreground
- Need to tune thresholds carefully

## Recommended Approach

### Phase 1: Quick Wins (Immediate)
1. **Enable foreground-biased cropping:**
   ```yaml
   foreground_crop_prob: 1.0
   min_foreground_ratio: 0.5
   ```

2. **Make local crops subsets:**
   ```yaml
   localcrops_subset_of_globalcrops: true
   ```

### Phase 2: Enhanced Matching (If Phase 1 helps but not enough)
3. **Add foreground-aware loss weighting:**
   - Weight losses by foreground ratio
   - Prioritize learning from meaningful content

### Phase 3: Advanced (If still needed)
4. **Hierarchical matching:**
   - Local-to-local for fine details
   - Local-to-global only when appropriate

## Implementation Notes

### Foreground Threshold
- Current: `foreground_threshold: 500` (HU units)
- Typical CT: Air = -1000 HU, Soft tissue = 0-100 HU, Bone = 200-3000 HU
- Threshold of 500 is quite high (mostly bone)
- Consider: `-500` (soft tissue and above) or `0` (all tissue)

### Foreground Ratio
- `min_foreground_ratio: 0.3` means 30% of voxels must be foreground
- For CT scans, 0.5-0.7 might be better (ensure meaningful content)
- Too high (0.9) might be too restrictive

### Local Crop Subset
- When `localcrops_subset_of_globalcrops: true`, local crops are sampled from within global crop
- This ensures they share spatial context
- Patch alignment is respected (crops align to patch boundaries)

## Testing Strategy

1. **Visualize crops:**
   - Check foreground ratios in actual crops
   - Verify local crops are within global crops (if enabled)
   - Ensure crops contain meaningful anatomical structures

2. **Monitor metrics:**
   - Track foreground ratios per crop
   - Monitor if entropy improves (less uniform)
   - Check if loss decreases more consistently

3. **Ablation studies:**
   - Try each solution independently
   - Combine solutions incrementally
   - Measure impact on uniform collapse

## Expected Outcomes

### If solutions work:
- **Entropy decreases** (less uniform, more structured)
- **Loss decreases more consistently** (no large increases)
- **Model learns anatomical features** (not just background)
- **Representations are more discriminative** (better downstream performance)

### If still not working:
- May need to adjust thresholds
- May need different loss formulation
- May need to reconsider architecture or data augmentation
