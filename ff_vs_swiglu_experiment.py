import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    d_model: int = 384
    d_ff: int = 1536
    batch_size: int = 32
    seq_len: int = 512
    num_iterations: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class StandardFeedForward(nn.Module):
    """Standard feedforward with SiLU activation"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class SwiGLU(nn.Module):
    """SwiGLU: Swish-Gated Linear Unit"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU uses 2/3 * d_ff for the gate and up projections
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate_output = F.silu(self.gate(x))  # Swish activation
        up_output = self.up(x)
        gated = gate_output * up_output  # Element-wise multiplication (gating)
        return self.down(self.dropout(gated))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_forward_pass(model, input_tensor, num_iterations):
    """Benchmark forward pass speed"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    return (end_time - start_time) / num_iterations

def benchmark_backward_pass(model, input_tensor, num_iterations):
    """Benchmark backward pass speed"""
    model.train()
    
    # Warmup
    for _ in range(10):
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    for _ in range(num_iterations):
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    return (end_time - start_time) / num_iterations

def run_experiment():
    config = ExperimentConfig()
    print(f"🧪 FeedForward vs SwiGLU Experiment")
    print(f"Device: {config.device}")
    print(f"Config: d_model={config.d_model}, d_ff={config.d_ff}")
    print(f"Input shape: [{config.batch_size}, {config.seq_len}, {config.d_model}]")
    print("-" * 60)
    
    # Create models
    ff_model = StandardFeedForward(config.d_model, config.d_ff).to(config.device)
    swiglu_model = SwiGLU(config.d_model, config.d_ff).to(config.device)
    
    # Create input tensor
    input_tensor = torch.randn(
        config.batch_size, config.seq_len, config.d_model, 
        device=config.device, requires_grad=True
    )
    
    # Count parameters
    ff_params = count_parameters(ff_model)
    swiglu_params = count_parameters(swiglu_model)
    
    print(f"📊 Parameter Count:")
    print(f"  Standard FF: {ff_params:,} parameters")
    print(f"  SwiGLU:      {swiglu_params:,} parameters")
    print(f"  Ratio:       {swiglu_params/ff_params:.2f}x more parameters")
    print()
    
    # Test output shapes
    with torch.no_grad():
        ff_output = ff_model(input_tensor)
        swiglu_output = swiglu_model(input_tensor)
    
    print(f"🔍 Output Shapes:")
    print(f"  Input:       {list(input_tensor.shape)}")
    print(f"  Standard FF: {list(ff_output.shape)}")
    print(f"  SwiGLU:      {list(swiglu_output.shape)}")
    print()
    
    # Benchmark forward pass
    print(f"⚡ Forward Pass Benchmark ({config.num_iterations} iterations):")
    ff_forward_time = benchmark_forward_pass(ff_model, input_tensor, config.num_iterations)
    swiglu_forward_time = benchmark_forward_pass(swiglu_model, input_tensor, config.num_iterations)
    
    print(f"  Standard FF: {ff_forward_time*1000:.3f} ms/iteration")
    print(f"  SwiGLU:      {swiglu_forward_time*1000:.3f} ms/iteration")
    print(f"  Speedup:     {swiglu_forward_time/ff_forward_time:.2f}x slower")
    print()
    
    # Benchmark backward pass
    print(f"🔄 Backward Pass Benchmark ({config.num_iterations} iterations):")
    ff_backward_time = benchmark_backward_pass(ff_model, input_tensor, config.num_iterations)
    swiglu_backward_time = benchmark_backward_pass(swiglu_model, input_tensor, config.num_iterations)
    
    print(f"  Standard FF: {ff_backward_time*1000:.3f} ms/iteration")
    print(f"  SwiGLU:      {swiglu_backward_time*1000:.3f} ms/iteration")
    print(f"  Speedup:     {swiglu_backward_time/ff_backward_time:.2f}x slower")
    print()
    
    # Memory usage comparison
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test FF memory
        _ = ff_model(input_tensor)
        ff_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test SwiGLU memory
        _ = swiglu_model(input_tensor)
        swiglu_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"💾 Memory Usage:")
        print(f"  Standard FF: {ff_memory:.1f} MB")
        print(f"  SwiGLU:      {swiglu_memory:.1f} MB")
        print(f"  Ratio:       {swiglu_memory/ff_memory:.2f}x more memory")
        print()
    
    # Quick quality test - check if outputs are reasonable
    print(f"🎯 Output Statistics:")
    with torch.no_grad():
        ff_stats = {
            'mean': ff_output.mean().item(),
            'std': ff_output.std().item(),
            'min': ff_output.min().item(),
            'max': ff_output.max().item()
        }
        swiglu_stats = {
            'mean': swiglu_output.mean().item(),
            'std': swiglu_output.std().item(),
            'min': swiglu_output.min().item(),
            'max': swiglu_output.max().item()
        }
    
    print(f"  Standard FF - Mean: {ff_stats['mean']:.4f}, Std: {ff_stats['std']:.4f}")
    print(f"  SwiGLU      - Mean: {swiglu_stats['mean']:.4f}, Std: {swiglu_stats['std']:.4f}")
    
    print("\n" + "="*60)
    print("🏁 SUMMARY:")
    print(f"  • SwiGLU has {swiglu_params/ff_params:.1f}x more parameters")
    print(f"  • SwiGLU is {swiglu_forward_time/ff_forward_time:.1f}x slower in forward pass")
    print(f"  • SwiGLU is {swiglu_backward_time/ff_backward_time:.1f}x slower in backward pass")
    if torch.cuda.is_available():
        print(f"  • SwiGLU uses {swiglu_memory/ff_memory:.1f}x more memory")
    print("  • SwiGLU typically provides better model quality despite overhead")

if __name__ == "__main__":
    run_experiment()