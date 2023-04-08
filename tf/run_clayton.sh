# Comparison with MCMC
python -u clayton.py --dim 8  --exp_sampling --exp_mcmc --exp_benchmark
python -u clayton.py --dim 16 --exp_sampling --exp_mcmc --exp_benchmark
python -u clayton.py --dim 32 --exp_sampling --exp_mcmc --exp_benchmark
python -u clayton.py --dim 64 --exp_sampling --exp_mcmc --exp_benchmark

python -u clayton.py --dim 8  --mcmc_thinning 10 --exp_mcmc
python -u clayton.py --dim 16 --mcmc_thinning 10 --exp_mcmc
python -u clayton.py --dim 32 --mcmc_thinning 10 --exp_mcmc
python -u clayton.py --dim 64 --mcmc_thinning 10 --exp_mcmc

# Different transport maps
python -u clayton_extra.py --dim 8  --alpha 0.7 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 16 --alpha 0.7 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 32 --alpha 0.7 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 64 --alpha 0.7 --spline linear --exp_sampling --exp_benchmark

python -u clayton_extra.py --dim 8  --alpha 0.7 --spline quadratic --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 16 --alpha 0.7 --spline quadratic --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 32 --alpha 0.7 --spline quadratic --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 64 --alpha 0.7 --spline quadratic --exp_sampling --exp_benchmark

# Different alpha values
python -u clayton_extra.py --dim 8  --alpha 0.5 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 16 --alpha 0.5 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 32 --alpha 0.5 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 64 --alpha 0.5 --spline linear --exp_sampling --exp_benchmark

python -u clayton_extra.py --dim 8  --alpha 0.6 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 16 --alpha 0.6 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 32 --alpha 0.6 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 64 --alpha 0.6 --spline linear --exp_sampling --exp_benchmark

python -u clayton_extra.py --dim 8  --alpha 0.8 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 16 --alpha 0.8 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 32 --alpha 0.8 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 64 --alpha 0.8 --spline linear --exp_sampling --exp_benchmark

python -u clayton_extra.py --dim 8  --alpha 0.9 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 16 --alpha 0.9 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 32 --alpha 0.9 --spline linear --exp_sampling --exp_benchmark
python -u clayton_extra.py --dim 64 --alpha 0.9 --spline linear --exp_sampling --exp_benchmark

# Experiment on importance sampling
python -u clayton_is.py
