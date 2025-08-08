#!/usr/bin/env python3
"""
Compare 7 different FeedForward architectures in actual LLM training:
1. Standard FF (ReLU)
2. SwiGLU 
3. GELU FF
4. Mish FF
5. GLU (Gated Linear Unit)
6. ReGLU (ReLU + GLU)
7. GeGLU (GELU + GLU)
"""

import torch
from llm import ModelConfig, load_and_cache_data, TextTokenDataset, train_model, set_seed, setup_muon_optimizer, evaluate_model, MinimalLLM
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import math
from tqdm import tqdm
import numpy as np

def train_model_with_history(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Modified train_model that returns loss history"""
    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Training loop with history tracking
    model.train()
    step = 0
    train_losses, val_losses, steps = [], [], []

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Store train loss every 100 steps
            if step % 100 == 0:
                train_losses.append(loss.item() * config.gradient_accumulation_steps)
                steps.append(step)

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                val_losses.append(eval_metrics['val_loss'])

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    return model, final_eval, train_losses, val_losses, steps

def run_comparison():
    print("🔬 LLM Training: FeedForward vs SwiGLU Comparison")
    print("=" * 60)
    
    # Set global seed for reproducibility
    set_seed(42)
    
    # Shared config
    base_config = ModelConfig(
        max_steps=4000,  # Extended training for better comparison
        eval_every=250,
        batch_size=16,   # Smaller batch for faster iteration
        num_documents=1000,  # Less data for speed
        max_tokens=250000
    )
    
    # Load data once
    print("📦 Loading data...")
    texts, tokenizer, tokens = load_and_cache_data(base_config)
    dataset = TextTokenDataset(tokens, base_config.max_seq_len)
    
    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=base_config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=base_config.batch_size, shuffle=False, num_workers=2)
    
    results = {}
    
    # Define all feedforward variants to test
    ff_variants = [
        ("Standard FF", {"use_swiglu": False, "ff_activation": "relu"}),
        ("SwiGLU", {"use_swiglu": True, "ff_activation": "silu"}),
        ("GELU FF", {"use_swiglu": False, "ff_activation": "gelu"}),
        ("Mish FF", {"use_swiglu": False, "ff_activation": "mish"}),
        ("GLU", {"use_swiglu": False, "ff_activation": "glu"}),
        ("ReGLU", {"use_swiglu": False, "ff_activation": "reglu"}),
        ("GeGLU", {"use_swiglu": False, "ff_activation": "geglu"}),
    ]
    
    # Test all configurations
    for ff_name, ff_config in ff_variants:
        # Reset seed before each training run for fair comparison
        set_seed(42)
        
        # Create new config with feedforward variant
        config_dict = base_config.__dict__.copy()
        # Remove computed fields that shouldn't be passed to constructor
        config_dict.pop('d_k', None)
        config_dict.update(ff_config)
        config = ModelConfig(**config_dict)
        
        print(f"\n{'='*20} {ff_name} {'='*20}")
        
        # Clear GPU cache before timing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        start_time = time.time()
        model, metrics, train_losses, val_losses, steps = train_model_with_history(config, train_loader, val_loader)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_time = time.time() - start_time
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        results[ff_name] = {
            'metrics': metrics,
            'training_time': training_time,
            'total_params': total_params,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'steps': steps,
            'model': model  # Store model for weight analysis
        }
        
        print(f"⏱️  Training time: {training_time/60:.1f} minutes")
        print(f"📊 Parameters: {total_params:,}")
        print(f"🎯 Final Loss: {metrics['val_loss']:.4f}, PPL: {metrics['val_perplexity']:.2f}")
    
    # Compare results
    print("\n" + "="*80)
    print("🏆 FEEDFORWARD ARCHITECTURE COMPARISON RESULTS")
    print("="*80)
    
    # Sort by validation loss (best first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['val_loss'])
    
    print(f"{'Rank':<4} {'Architecture':<12} {'Params':<10} {'Time(min)':<9} {'Val Loss':<9} {'Perplexity':<10}")
    print("-" * 70)
    
    baseline_loss = results["Standard FF"]['metrics']['val_loss']
    
    for rank, (name, result) in enumerate(sorted_results, 1):
        params = f"{result['total_params']/1e6:.1f}M"
        time_min = f"{result['training_time']/60:.1f}"
        val_loss = f"{result['metrics']['val_loss']:.4f}"
        ppl = f"{result['metrics']['val_perplexity']:.1f}"
        
        improvement = (baseline_loss - result['metrics']['val_loss']) / baseline_loss * 100
        improvement_str = f"({improvement:+.1f}%)" if name != "Standard FF" else "(baseline)"
        
        print(f"{rank:<4} {name:<12} {params:<10} {time_min:<9} {val_loss:<9} {ppl:<10} {improvement_str}")
    
    print(f"\n🏆 Best performing: {sorted_results[0][0]}")
    print(f"📈 Best improvement over Standard FF: {(baseline_loss - sorted_results[0][1]['metrics']['val_loss']) / baseline_loss * 100:+.1f}%")
    
    # Plot training curves
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    plt.figure(figsize=(16, 10))
    
    # Training loss
    plt.subplot(2, 2, 1)
    for i, (name, result) in enumerate(results.items()):
        plt.plot(result['steps'], result['train_losses'], color=colors[i], label=name, alpha=0.8, linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Validation loss
    plt.subplot(2, 2, 2)
    for i, (name, result) in enumerate(results.items()):
        val_steps = list(range(0, len(result['val_losses']) * base_config.eval_every, base_config.eval_every))
        if len(val_steps) > len(result['val_losses']):
            val_steps = val_steps[:len(result['val_losses'])]
        plt.plot(val_steps, result['val_losses'], color=colors[i], label=name, alpha=0.8, linewidth=1.5)
    plt.xlabel('Steps')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Final performance bar chart
    plt.subplot(2, 2, 3)
    names = list(results.keys())
    final_losses = [results[name]['metrics']['val_loss'] for name in names]
    bars = plt.bar(range(len(names)), final_losses, color=colors[:len(names)], alpha=0.7)
    plt.xlabel('Architecture')
    plt.ylabel('Final Validation Loss')
    plt.title('Final Performance Comparison')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{loss:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Weight distribution analysis
    plt.subplot(2, 2, 4)
    for i, (name, result) in enumerate(results.items()):
        model = result['model']
        # Get feedforward weights from first layer
        ff_weights = []
        for param_name, param in model.named_parameters():
            if 'ff' in param_name.lower() and 'weight' in param_name:
                ff_weights.extend(param.detach().cpu().flatten().numpy())
                break  # Just use first FF layer
        
        if ff_weights:
            plt.hist(ff_weights, bins=50, alpha=0.6, label=name, color=colors[i], density=True)
    
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.title('FF Weight Distributions (Layer 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ff_architectures_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Comprehensive plot saved as 'ff_architectures_comparison.png'")

if __name__ == "__main__":
    run_comparison()