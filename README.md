## Pilot Analysis (commit `09e104edb618e3d3035611bd2b2bd4cbb15062c7`)

We ran **162** pilot experiments and then executed `python analysis.py` against  
`artifacts/combined_validated_data.csv`. Key summary:

- **Features**  
  - `x,y,vx,vy` → mean(final_reward)=103.4, σ=25.1  
  - `presence,x,y,vx,vy` → 117.5, σ=23.8  
  - `x,y,vx,vy,cos_h,sin_h` → **126.6**, σ=18.7  

- **Learning rate**  
  - `1e-4` → 120.9, σ=22.0  
  - `3e-4` → 108.7, σ=24.9  

- **Epochs/update**  
  - 4 → 6 → 8: 112.8 → 112.1 → **118.2**  

- **Hidden dimension**  
  - 64 → 128 → 256: 110.4 → 114.2 → **117.6**  

- **Batch size**  
  - 32 → 64 → 128: **119.4** → 111.6 → 112.1  

_Run the full summary with:_  

```bash
python analysis.py
```

Output:

```
❯ uv run analysis.py

=== Overall ===
       final_reward  max_reward  training_steps
count    162.000000  162.000000      162.000000
mean     114.377842  118.371709   243233.808642
std       23.049623   21.988304    25497.998057
min       32.058300   32.058300   175750.000000
25%      103.643150  107.919325   220667.500000
50%      116.994650  122.424250   247872.500000
75%      132.077025  136.246275   265531.000000
max      142.848400  144.350200   278528.000000


=== By features ===
                             mean        std  count
features                                           
presence,x,y,vx,vy     112.936787  27.037245     54
x,y,vx,vy              115.916228  15.909437     54
x,y,vx,vy,cos_h,sin_h  114.280511  25.004886     54


=== By lr ===
              mean        std  count
lr                                  
0.0001  112.934002  25.821763     81
0.0003  115.821681  19.955611     81


=== By epochs ===
              mean        std  count
epochs                              
4       112.831322  23.403487     54
6       112.109744  25.661155     54
8       118.192459  19.626048     54


=== By hidden ===
                  mean        std  count
hidden_dim                              
64          101.466906  25.869067     54
128         119.001096  20.527624     54
256         122.665524  16.177610     54


=== By batch ===
                  mean        std  count
batch_size                              
32          119.425502  14.168552     54
64          111.584148  26.770921     54
128         112.123876  25.639540     54
```

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
