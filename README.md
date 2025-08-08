# FeedForward vs SwiGLU Experiment

## Overview
Minimal comparison between standard feedforward layers and SwiGLU activation in transformer architectures.

## What's Being Compared

**Standard FeedForward:**
```python
def forward(x):
    return linear2(dropout(silu(linear1(x))))
```

**SwiGLU:**
```python
def forward(x):
    gate = silu(gate_proj(x))
    up = up_proj(x)
    return down_proj(dropout(gate * up))
```

## Fair Comparison

- **Same d_model**: Both input/output 384 dimensions
- **Same d_ff**: Both use 1536 hidden dimensions  
- **Same activation**: Both use SiLU/Swish
- **Same dropout**: Applied consistently
- **Same initialization**: Standard PyTorch defaults

## Key Differences

- **Parameters**: SwiGLU has 1.5x more parameters (needs separate gate + up projections)
- **Computation**: SwiGLU does element-wise gating (multiplication)
- **Memory**: SwiGLU uses more intermediate activations

## Running

```bash
python ff_vs_swiglu_experiment.py
```

Benchmarks forward/backward speed, memory usage, and parameter count.