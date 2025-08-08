#!/usr/bin/env python3
"""
Compare FeedForward vs SwiGLU in actual LLM training
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
        max_steps=2000,  # Increased for better comparison
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
    
    # Test both configurations
    for use_swiglu in [False, True]:
        # Reset seed before each training run for fair comparison
        set_seed(42)
        
        # Create new config with same parameters but different use_swiglu
        config = ModelConfig(
            d_model=base_config.d_model,
            n_heads=base_config.n_heads,
            n_layers=base_config.n_layers,
            d_ff=base_config.d_ff,
            batch_size=base_config.batch_size,
            max_steps=base_config.max_steps,
            use_swiglu=use_swiglu,
            gradient_accumulation_steps=base_config.gradient_accumulation_steps,
            muon_lr=base_config.muon_lr,
            max_seq_len=base_config.max_seq_len,
            num_documents=base_config.num_documents,
            max_tokens=base_config.max_tokens,
            eval_every=base_config.eval_every,
            eval_steps=base_config.eval_steps,
            weight_decay=base_config.weight_decay,
            dropout=base_config.dropout,
            grad_clip=base_config.grad_clip,
            use_amp=base_config.use_amp,
            vocab_size=base_config.vocab_size
        )
        
        ff_type = "SwiGLU" if use_swiglu else "Standard FF"
        print(f"\n{'='*20} {ff_type} {'='*20}")
        
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
        
        results[ff_type] = {
            'metrics': metrics,
            'training_time': training_time,
            'total_params': total_params,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'steps': steps
        }
        
        print(f"⏱️  Training time: {training_time/60:.1f} minutes")
        print(f"📊 Parameters: {total_params:,}")
    
    # Compare results
    print("\n" + "="*60)
    print("🏆 COMPARISON RESULTS")
    print("="*60)
    
    ff_result = results["Standard FF"]
    swiglu_result = results["SwiGLU"]
    
    print(f"📊 Parameters:")
    print(f"  Standard FF: {ff_result['total_params']:,}")
    print(f"  SwiGLU:      {swiglu_result['total_params']:,}")
    print(f"  Ratio:       {swiglu_result['total_params']/ff_result['total_params']:.2f}x")
    
    print(f"\n⏱️  Training Time:")
    print(f"  Standard FF: {ff_result['training_time']/60:.1f} min")
    print(f"  SwiGLU:      {swiglu_result['training_time']/60:.1f} min")
    print(f"  Ratio:       {swiglu_result['training_time']/ff_result['training_time']:.2f}x")
    
    print(f"\n🎯 Final Performance:")
    print(f"  Standard FF - Loss: {ff_result['metrics']['val_loss']:.4f}, PPL: {ff_result['metrics']['val_perplexity']:.2f}")
    print(f"  SwiGLU      - Loss: {swiglu_result['metrics']['val_loss']:.4f}, PPL: {swiglu_result['metrics']['val_perplexity']:.2f}")
    
    loss_improvement = (ff_result['metrics']['val_loss'] - swiglu_result['metrics']['val_loss']) / ff_result['metrics']['val_loss'] * 100
    ppl_improvement = (ff_result['metrics']['val_perplexity'] - swiglu_result['metrics']['val_perplexity']) / ff_result['metrics']['val_perplexity'] * 100
    
    print(f"\n📈 SwiGLU vs Standard FF:")
    print(f"  Loss improvement: {loss_improvement:+.1f}%")
    print(f"  PPL improvement:  {ppl_improvement:+.1f}%")
    
    if loss_improvement > 0:
        print(f"  ✅ SwiGLU achieved better loss despite {swiglu_result['total_params']/ff_result['total_params']:.1f}x more parameters")
    else:
        print(f"  ❌ SwiGLU performed worse despite {swiglu_result['total_params']/ff_result['total_params']:.1f}x more parameters")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Training loss
    plt.subplot(1, 2, 1)
    plt.plot(ff_result['steps'], ff_result['train_losses'], 'b-', label='Standard FF', alpha=0.7)
    plt.plot(swiglu_result['steps'], swiglu_result['train_losses'], 'r-', label='SwiGLU', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation loss
    plt.subplot(1, 2, 2)
    val_steps = list(range(0, len(ff_result['val_losses']) * base_config.eval_every, base_config.eval_every))
    if len(val_steps) > len(ff_result['val_losses']):
        val_steps = val_steps[:len(ff_result['val_losses'])]
    plt.plot(val_steps, ff_result['val_losses'], 'b-', label='Standard FF', alpha=0.7)
    plt.plot(val_steps, swiglu_result['val_losses'], 'r-', label='SwiGLU', alpha=0.7)
    plt.xlabel('Steps')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ff_vs_swiglu_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Plot saved as 'ff_vs_swiglu_comparison.png'")

if __name__ == "__main__":
    run_comparison()