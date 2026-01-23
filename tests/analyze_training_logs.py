"""
Analyze training logs to determine if the DINOv3 3D model is learning.

This script:
1. Parses training_metrics.json (JSONL format)
2. Plots loss curves and key metrics
3. Analyzes entropy metrics to detect collapse
4. Analyzes gradient norms to check training dynamics
5. Provides insights on whether the model is learning meaningful representations
"""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_metrics(jsonl_path):
    """Load metrics from JSONL file."""
    metrics = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def extract_metrics(metrics_list):
    """Extract key metrics into arrays for plotting."""
    data = defaultdict(list)
    
    for m in metrics_list:
        data['iteration'].append(m.get('iteration', 0))
        data['total_loss'].append(m.get('total_loss', 0))
        data['dino_global_crops_loss'].append(m.get('dino_global_crops_loss', 0))
        data['dino_local_crops_loss'].append(m.get('dino_local_crops_loss', 0))
        data['dino_local_loss_weight'].append(m.get('dino_local_loss_weight', 0))
        data['ibot_loss'].append(m.get('ibot_loss', 0))
        data['koleo_loss'].append(m.get('koleo_loss', 0))
        data['lr'].append(m.get('lr', 0))
        data['backbone_grad_norm_after_bp'].append(m.get('backbone_grad_norm_after_bp', 0))
        data['head_grad_norm_after_bp'].append(m.get('head_grad_norm_after_bp', 0))
        data['dino_teacher_entropy'].append(m.get('dino_teacher_entropy', 0))
        data['dino_student_entropy'].append(m.get('dino_student_entropy', 0))
        data['dino_teacher_entropy_vs_uniform'].append(m.get('dino_teacher_entropy_vs_uniform', 0))
        data['dino_student_entropy_vs_uniform'].append(m.get('dino_student_entropy_vs_uniform', 0))
        data['dino_teacher_max_prob'].append(m.get('dino_teacher_max_prob', 0))
        data['dino_student_max_prob'].append(m.get('dino_student_max_prob', 0))
        data['dino_teacher_logits_std'].append(m.get('dino_teacher_logits_std', 0))
        data['dino_teacher_logits_range'].append(m.get('dino_teacher_logits_range', 0))
        data['dino_teacher_crop_cross_entropy'].append(m.get('dino_teacher_crop_cross_entropy', 0))
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def analyze_training(data):
    """Analyze training dynamics and provide insights."""
    print("=" * 80)
    print("TRAINING ANALYSIS")
    print("=" * 80)
    
    iterations = data['iteration']
    total_loss = data['total_loss']
    
    # Basic statistics
    print(f"\n1. BASIC STATISTICS")
    print(f"   Total iterations: {len(iterations)}")
    print(f"   Iteration range: {iterations[0]} - {iterations[-1]}")
    print(f"   Initial loss: {total_loss[0]:.4f}")
    print(f"   Final loss: {total_loss[-1]:.4f}")
    print(f"   Loss change: {total_loss[-1] - total_loss[0]:.4f}")
    print(f"   Min loss: {total_loss.min():.4f} (at iteration {iterations[total_loss.argmin()]})")
    print(f"   Max loss: {total_loss.max():.4f} (at iteration {iterations[total_loss.argmax()]})")
    
    # Loss trajectory analysis
    print(f"\n2. LOSS TRAJECTORY")
    # Find where loss drops to minimum
    min_idx = total_loss.argmin()
    min_iter = iterations[min_idx]
    min_loss = total_loss[min_idx]
    print(f"   Loss minimum: {min_loss:.4f} at iteration {min_iter}")
    
    # Check if loss increased after minimum
    if min_idx < len(total_loss) - 1:
        final_loss = total_loss[-1]
        loss_increase = final_loss - min_loss
        print(f"   Loss after minimum: {final_loss:.4f}")
        print(f"   Loss increase from min: {loss_increase:.4f}")
        if loss_increase > 0.5:
            print(f"   ⚠️  WARNING: Loss increased significantly after minimum!")
    
    # Entropy analysis (collapse detection)
    print(f"\n3. ENTROPY ANALYSIS (Collapse Detection)")
    teacher_entropy = data['dino_teacher_entropy']
    student_entropy = data['dino_student_entropy']
    teacher_entropy_vs_uniform = data['dino_teacher_entropy_vs_uniform']
    student_entropy_vs_uniform = data['dino_student_entropy_vs_uniform']
    
    # For 1024 prototypes, uniform entropy = log(1024) ≈ 6.93
    uniform_entropy = np.log(1024)
    print(f"   Uniform distribution entropy: {uniform_entropy:.4f}")
    print(f"   Teacher entropy - initial: {teacher_entropy[0]:.4f}, final: {teacher_entropy[-1]:.4f}")
    print(f"   Student entropy - initial: {student_entropy[0]:.4f}, final: {student_entropy[-1]:.4f}")
    print(f"   Teacher entropy vs uniform - initial: {teacher_entropy_vs_uniform[0]:.4f}, final: {teacher_entropy_vs_uniform[-1]:.4f}")
    print(f"   Student entropy vs uniform - initial: {student_entropy_vs_uniform[0]:.4f}, final: {student_entropy_vs_uniform[-1]:.4f}")
    
    # Check for collapse (entropy too low OR too high/uniform)
    # Uniform collapse: entropy too close to uniform (ratio > 0.95)
    # Mode collapse: entropy too low (ratio < 0.5)
    teacher_entropy_ratio = teacher_entropy_vs_uniform[-1]
    student_entropy_ratio = student_entropy_vs_uniform[-1]
    
    if teacher_entropy[-1] < uniform_entropy * 0.5:
        print(f"   ⚠️  WARNING: Teacher entropy is very low - MODE COLLAPSE!")
    elif teacher_entropy[-1] < uniform_entropy * 0.7:
        print(f"   ⚠️  CAUTION: Teacher entropy is moderately low")
    elif teacher_entropy_ratio > 0.95:
        print(f"   ⚠️  WARNING: Teacher entropy is almost UNIFORM (ratio={teacher_entropy_ratio:.4f}) - UNIFORM COLLAPSE!")
        print(f"      This means the model is not learning to distinguish features!")
    elif teacher_entropy_ratio > 0.90:
        print(f"   ⚠️  CAUTION: Teacher entropy is very close to uniform (ratio={teacher_entropy_ratio:.4f})")
    else:
        print(f"   ✓ Teacher entropy looks healthy (ratio={teacher_entropy_ratio:.4f})")
    
    if student_entropy[-1] < uniform_entropy * 0.5:
        print(f"   ⚠️  WARNING: Student entropy is very low - MODE COLLAPSE!")
    elif student_entropy[-1] < uniform_entropy * 0.7:
        print(f"   ⚠️  CAUTION: Student entropy is moderately low")
    elif student_entropy_ratio > 0.95:
        print(f"   ⚠️  WARNING: Student entropy is almost UNIFORM (ratio={student_entropy_ratio:.4f}) - UNIFORM COLLAPSE!")
        print(f"      This means the model is not learning to distinguish features!")
    elif student_entropy_ratio > 0.90:
        print(f"   ⚠️  CAUTION: Student entropy is very close to uniform (ratio={student_entropy_ratio:.4f})")
    else:
        print(f"   ✓ Student entropy looks healthy (ratio={student_entropy_ratio:.4f})")
    
    # Max probability analysis
    teacher_max_prob = data['dino_teacher_max_prob']
    student_max_prob = data['dino_student_max_prob']
    uniform_max_prob = 1.0 / 1024  # For 1024 prototypes
    print(f"\n4. MAX PROBABILITY ANALYSIS")
    print(f"   Uniform max prob: {uniform_max_prob:.6f}")
    print(f"   Teacher max prob - initial: {teacher_max_prob[0]:.6f}, final: {teacher_max_prob[-1]:.6f}")
    print(f"   Student max prob - initial: {student_max_prob[0]:.6f}, final: {student_max_prob[-1]:.6f}")
    print(f"   Teacher max prob vs uniform - initial: {teacher_max_prob[0]/uniform_max_prob:.2f}x, final: {teacher_max_prob[-1]/uniform_max_prob:.2f}x")
    
    # Check for uniform collapse: max prob too close to uniform
    if teacher_max_prob[-1] < uniform_max_prob * 1.5:
        print(f"   ⚠️  WARNING: Teacher max prob is very close to uniform ({teacher_max_prob[-1]/uniform_max_prob:.2f}x) - UNIFORM COLLAPSE!")
        print(f"      The model is not learning to be confident about any prototype!")
    elif teacher_max_prob[-1] > uniform_max_prob * 10:
        print(f"   ⚠️  WARNING: Teacher is too confident (high max prob) - MODE COLLAPSE!")
    elif teacher_max_prob[-1] < uniform_max_prob * 3.0:
        print(f"   ⚠️  CAUTION: Teacher max prob is close to uniform ({teacher_max_prob[-1]/uniform_max_prob:.2f}x)")
    else:
        print(f"   ✓ Max probabilities look reasonable")
    
    # Gradient analysis
    print(f"\n5. GRADIENT ANALYSIS")
    backbone_grad = data['backbone_grad_norm_after_bp']
    head_grad = data['head_grad_norm_after_bp']
    print(f"   Backbone grad norm - initial: {backbone_grad[0]:.4f}, final: {backbone_grad[-1]:.4f}")
    print(f"   Head grad norm - initial: {head_grad[0]:.4f}, final: {head_grad[-1]:.4f}")
    print(f"   Backbone grad norm - mean: {backbone_grad.mean():.4f}, std: {backbone_grad.std():.4f}")
    print(f"   Head grad norm - mean: {head_grad.mean():.4f}, std: {head_grad.std():.4f}")
    
    if backbone_grad[-1] < 0.1:
        print(f"   ⚠️  WARNING: Backbone gradients are very small - model may have stopped learning!")
    elif backbone_grad[-1] < 1.0:
        print(f"   ⚠️  CAUTION: Backbone gradients are small")
    else:
        print(f"   ✓ Backbone gradients look healthy")
    
    # Learning rate analysis
    print(f"\n6. LEARNING RATE ANALYSIS")
    lr = data['lr']
    print(f"   Initial LR: {lr[0]:.6f}")
    print(f"   Final LR: {lr[-1]:.6f}")
    print(f"   Max LR: {lr.max():.6f} (at iteration {iterations[lr.argmax()]})")
    
    # Loss components
    print(f"\n7. LOSS COMPONENTS")
    dino_global = data['dino_global_crops_loss']
    dino_local = data['dino_local_crops_loss']
    ibot = data['ibot_loss']
    koleo = data['koleo_loss']
    print(f"   DINO global loss - initial: {dino_global[0]:.4f}, final: {dino_global[-1]:.4f}")
    print(f"   DINO local loss - initial: {dino_local[0]:.4f}, final: {dino_local[-1]:.4f}")
    print(f"   iBOT loss - initial: {ibot[0]:.4f}, final: {ibot[-1]:.4f}")
    print(f"   KoLeo loss - initial: {koleo[0]:.4f}, final: {koleo[-1]:.4f}")
    
    # Teacher-student agreement
    print(f"\n8. TEACHER-STUDENT AGREEMENT")
    teacher_crop_ce = data['dino_teacher_crop_cross_entropy']
    print(f"   Teacher crop cross-entropy (lower = more agreement) - initial: {teacher_crop_ce[0]:.4f}, final: {teacher_crop_ce[-1]:.4f}")
    if teacher_crop_ce[-1] < teacher_crop_ce[0]:
        print(f"   ✓ Teacher crops are becoming more similar (good for learning)")
    else:
        print(f"   ⚠️  Teacher crops are becoming less similar")
    
    # Logits std/range analysis (uniform detection)
    print(f"\n9. LOGITS ANALYSIS (Uniform Detection)")
    teacher_logits_std = data['dino_teacher_logits_std']
    teacher_logits_range = data['dino_teacher_logits_range']
    print(f"   Teacher logits std - initial: {teacher_logits_std[0]:.4f}, final: {teacher_logits_std[-1]:.4f}")
    print(f"   Teacher logits range - initial: {teacher_logits_range[0]:.4f}, final: {teacher_logits_range[-1]:.4f}")
    
    # If logits std/range is very small, outputs are uniform
    if teacher_logits_std[-1] < 0.05:
        print(f"   ⚠️  WARNING: Teacher logits std is very small ({teacher_logits_std[-1]:.4f}) - outputs are UNIFORM!")
    elif teacher_logits_std[-1] < 0.1:
        print(f"   ⚠️  CAUTION: Teacher logits std is small ({teacher_logits_std[-1]:.4f}) - outputs may be too uniform")
    else:
        print(f"   ✓ Teacher logits have reasonable variance")
    
    if teacher_logits_range[-1] < 0.2:
        print(f"   ⚠️  WARNING: Teacher logits range is very small ({teacher_logits_range[-1]:.4f}) - outputs are UNIFORM!")
    elif teacher_logits_range[-1] < 0.5:
        print(f"   ⚠️  CAUTION: Teacher logits range is small ({teacher_logits_range[-1]:.4f})")
    else:
        print(f"   ✓ Teacher logits have reasonable range")
    
    # Overall assessment
    print(f"\n10. OVERALL ASSESSMENT")
    print("=" * 80)
    
    # Check various indicators
    indicators = []
    uniform_collapse_detected = False
    
    # Loss decreased from start
    if total_loss[-1] < total_loss[0]:
        indicators.append("✓ Loss decreased from start")
    else:
        indicators.append("✗ Loss did not decrease from start")
    
    # Check for uniform collapse
    teacher_entropy_ratio = teacher_entropy_vs_uniform[-1]
    student_entropy_ratio = student_entropy_vs_uniform[-1]
    if teacher_entropy_ratio > 0.95 or student_entropy_ratio > 0.95:
        indicators.append("✗ UNIFORM COLLAPSE DETECTED (entropy too close to uniform)")
        uniform_collapse_detected = True
    elif teacher_entropy_ratio > 0.90 or student_entropy_ratio > 0.90:
        indicators.append("⚠️  Entropy very close to uniform (possible uniform collapse)")
        uniform_collapse_detected = True
    elif teacher_entropy[-1] > uniform_entropy * 0.7 and student_entropy[-1] > uniform_entropy * 0.7:
        indicators.append("✓ Entropy is healthy (no collapse)")
    else:
        indicators.append("✗ Entropy is low (possible mode collapse)")
    
    # Check logits for uniform collapse
    if teacher_logits_std[-1] < 0.05 or teacher_logits_range[-1] < 0.2:
        indicators.append("✗ Logits are uniform (uniform collapse)")
        uniform_collapse_detected = True
    else:
        indicators.append("✓ Logits have variance (not uniform)")
    
    # Max prob check for uniform collapse
    if teacher_max_prob[-1] < uniform_max_prob * 1.5:
        indicators.append("✗ Max prob too close to uniform (uniform collapse)")
        uniform_collapse_detected = True
    elif teacher_max_prob[-1] < uniform_max_prob * 3.0:
        indicators.append("⚠️  Max prob close to uniform")
    elif teacher_max_prob[-1] < uniform_max_prob * 10:
        indicators.append("✓ Max probability is reasonable")
    else:
        indicators.append("✗ Max probability is too high (mode collapse)")
    
    # Gradients are flowing
    if backbone_grad[-1] > 0.1:
        indicators.append("✓ Gradients are flowing")
    else:
        indicators.append("✗ Gradients are very small")
    
    for indicator in indicators:
        print(f"   {indicator}")
    
    # Final verdict
    print(f"\n11. VERDICT")
    print("=" * 80)
    
    positive_indicators = sum(1 for i in indicators if i.startswith("✓"))
    total_indicators = len(indicators)
    
    if uniform_collapse_detected:
        print("   ✗ UNIFORM COLLAPSE DETECTED - MODEL IS NOT LEARNING")
        print("   The model's outputs are becoming uniform across all prototypes.")
        print("   This means the model cannot distinguish between different features.")
        print("   Possible causes:")
        print("   - Learning rate too high or too low")
        print("   - Teacher temperature too high")
        print("   - Insufficient regularization (KoLeo loss may help)")
        print("   - Dataset too small or not diverse enough")
        print("   - Model architecture issues")
    elif positive_indicators >= 3:
        print("   ✓ MODEL APPEARS TO BE LEARNING")
        print("   The model shows signs of learning with:")
        print("   - Reasonable entropy (no collapse)")
        print("   - Active gradients")
        print("   - Loss trajectory showing learning")
    elif positive_indicators >= 2:
        print("   ⚠️  MODEL MAY BE LEARNING BUT WITH CONCERNS")
        print("   Some indicators are positive but others suggest issues.")
    else:
        print("   ✗ MODEL MAY NOT BE LEARNING EFFECTIVELY")
        print("   Multiple indicators suggest the model is not learning well.")
    
    print("=" * 80)


def plot_metrics(data, output_dir):
    """Create comprehensive plots of training metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    iterations = data['iteration']
    
    # 1. Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(iterations, data['total_loss'], label='Total Loss', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss Over Training')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Loss components
    ax = axes[0, 1]
    ax.plot(iterations, data['dino_global_crops_loss'], label='DINO Global', alpha=0.7)
    ax.plot(iterations, data['dino_local_crops_loss'], label='DINO Local', alpha=0.7)
    ax.plot(iterations, data['ibot_loss'], label='iBOT', alpha=0.7)
    ax.plot(iterations, data['koleo_loss'], label='KoLeo', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Entropy
    ax = axes[1, 0]
    uniform_entropy = np.log(1024)
    ax.axhline(y=uniform_entropy, color='r', linestyle='--', label='Uniform Entropy', alpha=0.5)
    ax.axhline(y=uniform_entropy * 0.95, color='orange', linestyle=':', label='95% Uniform (Collapse)', alpha=0.5)
    ax.plot(iterations, data['dino_teacher_entropy'], label='Teacher Entropy', linewidth=2)
    ax.plot(iterations, data['dino_student_entropy'], label='Student Entropy', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Over Training (Uniform Collapse Detection)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Gradient norms
    ax = axes[1, 1]
    ax.plot(iterations, data['backbone_grad_norm_after_bp'], label='Backbone Grad Norm', linewidth=2)
    ax.plot(iterations, data['head_grad_norm_after_bp'], label='Head Grad Norm', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_and_training_metrics.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_dir / 'loss_and_training_metrics.png'}")
    plt.close()
    
    # 2. Detailed loss analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss with moving average
    ax = axes[0, 0]
    window = min(100, len(iterations) // 10)
    if window > 1:
        moving_avg = np.convolve(data['total_loss'], np.ones(window)/window, mode='valid')
        moving_iter = iterations[window-1:]
        ax.plot(iterations, data['total_loss'], alpha=0.3, label='Raw')
        ax.plot(moving_iter, moving_avg, label=f'Moving Avg (window={window})', linewidth=2)
    else:
        ax.plot(iterations, data['total_loss'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss with Moving Average')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Learning rate
    ax = axes[0, 1]
    ax.plot(iterations, data['lr'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    # Max probabilities
    ax = axes[1, 0]
    uniform_max_prob = 1.0 / 1024
    collapse_threshold = uniform_max_prob * 1.5
    ax.axhline(y=uniform_max_prob, color='r', linestyle='--', label='Uniform Max Prob', alpha=0.5)
    ax.axhline(y=collapse_threshold, color='orange', linestyle=':', label='Collapse Threshold (1.5x)', alpha=0.5)
    ax.plot(iterations, data['dino_teacher_max_prob'], label='Teacher Max Prob', linewidth=2)
    ax.plot(iterations, data['dino_student_max_prob'], label='Student Max Prob', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Probability')
    ax.set_title('Max Probability (Uniform Collapse Detection)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Teacher crop agreement
    ax = axes[1, 1]
    ax.plot(iterations, data['dino_teacher_crop_cross_entropy'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cross-Entropy')
    ax.set_title('Teacher Crop Cross-Entropy (Lower = More Agreement)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_dir / 'detailed_analysis.png'}")
    plt.close()
    
    # 3. Uniform collapse detection plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Entropy ratio (vs uniform)
    ax = axes[0, 0]
    ax.axhline(y=1.0, color='r', linestyle='--', label='Uniform (1.0)', linewidth=2, alpha=0.7)
    ax.axhline(y=0.95, color='orange', linestyle=':', label='95% Uniform (Collapse)', alpha=0.7)
    ax.plot(iterations, data['dino_teacher_entropy_vs_uniform'], label='Teacher Entropy Ratio', linewidth=2)
    ax.plot(iterations, data['dino_student_entropy_vs_uniform'], label='Student Entropy Ratio', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Entropy Ratio (vs Uniform)')
    ax.set_title('Entropy Ratio - Uniform Collapse Detection')
    ax.set_ylim([0.5, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Logits std
    ax = axes[0, 1]
    ax.axhline(y=0.05, color='orange', linestyle=':', label='Collapse Threshold', alpha=0.7)
    ax.plot(iterations, data['dino_teacher_logits_std'], label='Teacher Logits Std', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Logits Standard Deviation')
    ax.set_title('Teacher Logits Std (Uniform Detection)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Logits range
    ax = axes[1, 0]
    ax.axhline(y=0.2, color='orange', linestyle=':', label='Collapse Threshold', alpha=0.7)
    ax.plot(iterations, data['dino_teacher_logits_range'], label='Teacher Logits Range', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Logits Range')
    ax.set_title('Teacher Logits Range (Uniform Detection)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Max prob ratio
    ax = axes[1, 1]
    uniform_max_prob = 1.0 / 1024
    max_prob_ratio = data['dino_teacher_max_prob'] / uniform_max_prob
    ax.axhline(y=1.0, color='r', linestyle='--', label='Uniform (1.0x)', linewidth=2, alpha=0.7)
    ax.axhline(y=1.5, color='orange', linestyle=':', label='Collapse Threshold (1.5x)', alpha=0.7)
    ax.plot(iterations, max_prob_ratio, label='Teacher Max Prob Ratio', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Prob Ratio (vs Uniform)')
    ax.set_title('Max Probability Ratio - Uniform Collapse Detection')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uniform_collapse_detection.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_dir / 'uniform_collapse_detection.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze DINOv3 3D training logs')
    parser.add_argument('--metrics-file', type=str, 
                       default='work_dir/training/training_metrics.json',
                       help='Path to training_metrics.json file')
    parser.add_argument('--output-dir', type=str,
                       default='work_dir/training/analysis',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    print(f"Loading metrics from: {args.metrics_file}")
    metrics = load_metrics(args.metrics_file)
    print(f"Loaded {len(metrics)} metric entries")
    
    data = extract_metrics(metrics)
    
    analyze_training(data)
    
    print(f"\nGenerating plots...")
    plot_metrics(data, args.output_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
