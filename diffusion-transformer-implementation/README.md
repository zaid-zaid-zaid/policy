This is a working implementation of a Diffusion Tranformer with EDM sampling in JAX. It also includes capabilities for experiment tracking with wandb. 
In my development and testing I focused on MNIST data. There is still a little bit left to do in terms of getting this implementation to work with D4RL data 
in a way that is aligned with the PGD codebase. 

To run experiments simply run code as written below.

```
cd experiments/mnist
python main.py
  --config=config.py
  --workdir=<dir>
  (--usewand)
```
