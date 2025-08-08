# FeedForward vs SwiGLU in LLM Training

Minimal experiment comparing standard feedforward vs SwiGLU in transformer training.

## Usage

```bash
python compare_ff_swiglu.py
```

Trains identical models with both feedforward types and compares performance.

## Implementation

- Standard FF: `linear2(dropout(silu(linear1(x))))`
- SwiGLU: `down(dropout(silu(gate(x)) * up(x)))`
- Fair comparison: same architecture, data, training setup
- SwiGLU has 1.5x parameters but often better quality

Part of [llm-research](https://github.com/vukrosic/llm-research)