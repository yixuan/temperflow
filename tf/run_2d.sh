# Comparison with MCMC
python -u 2d.py --data circle --exp_sampling --exp_mcmc
python -u 2d.py --data grid --exp_sampling --exp_mcmc
python -u 2d.py --data cross --exp_sampling --exp_mcmc

# Ablation experiments
python -u 2d.py --data circle --exp_ablation
python -u 2d.py --data grid --exp_ablation
python -u 2d.py --data cross --exp_ablation

# Truncated normal mixture
python -u 2d_truncated.py

# Different transport maps
python -u 2d_maps.py --data circle --bijector affine --exp_sampling
python -u 2d_maps.py --data circle --bijector lrspline --exp_sampling
python -u 2d_maps.py --data circle --bijector qrspline --exp_sampling

python -u 2d_maps.py --data grid --bijector affine --exp_sampling
python -u 2d_maps.py --data grid --bijector lrspline --exp_sampling
python -u 2d_maps.py --data grid --bijector qrspline --exp_sampling

python -u 2d_maps.py --data cross --bijector affine --exp_sampling
python -u 2d_maps.py --data cross --bijector lrspline --exp_sampling
python -u 2d_maps.py --data cross --bijector qrspline --exp_sampling
