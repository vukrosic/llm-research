#!/usr/bin/env python3
"""
Compare FeedForward vs SwiGLU in actual LLM training
"""

import torch
from llm import ModelConfig, load_and_cache_data, TextTokenDataset, train_model
from torch.utils.data import DataLoader
import time

def run_comparison():
    print("🔬 LLM Training: FeedForward vs SwiGLU Comparison")
    print("=" * 60)
    
    # Shared config - reduce steps for faster comparison
    base_config = ModelConfig(
        max_steps=1000,  # Reduced for quick comparison
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
        config = ModelConfig(**base_config.__dict__)
        config.use_swiglu = use_swiglu
        
        ff_type = "SwiGLU" if use_swiglu else "Standard FF"
        print(f"\n{'='*20} {ff_type} {'='*20}")
        
        start_time = time.time()
        model, metrics = train_model(config, train_loader, val_loader)
        training_time = time.time() - start_time
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        results[ff_type] = {
            'metrics': metrics,
            'training_time': training_time,
            'total_params': total_params
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

if __name__ == "__main__":
    run_comparison()