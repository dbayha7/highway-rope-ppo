## Midterm Pilot Analysis
>
> Data from commit `09e104edb618e3d3035611bd2b2bd4cbb15062c7`, `artifacts/combined_validated_data.csv`

Run:

```bash
python analysis.py
```

#### Key summaries (final_reward means ±σ)

- **Features**
  - `x,y,vx,vy` 115.92 ± 15.91
  - `presence,x,y,vx,vy` 112.94 ± 27.04
  - `x,y,vx,vy,cos_h,sin_h` 114.28 ± 25.00

- **Learning Rate**
  - `3e-4` 115.82 ± 19.96  ← **winner**  
  - `1e-4` 112.93 ± 25.82

- **Epochs/update**
  - 4 → 6 → **8** 112.83 → 112.11 → 118.19

- **Hidden Dim**
  - 64 → 128 → **256** 101.47 → 119.00 → 122.67

- **Batch Size**
  - **32** → 64 → 128 119.43 → 111.58 → 112.12

- **Clip ε**
  - 0.1 → **0.2**

- **Entropy Coef**
  - 0.001 → **0.005**
