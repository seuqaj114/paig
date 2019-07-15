# Physics-as-Inverse-Graphics

This repo contains the code for the paper Physics-as-Inverse-Graphics: Joint Unsupervised Learning of Objects and Physics from Video (https://arxiv.org/abs/1905.11169).

## Running experiments

To train run:

```
PYTHONPATH=. python runners/run_physics.py --task=spring_color --model=PhysicsNet --epochs=500 
--batch_size=100 --save_dir=<experiment_folder> --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true
--color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
```

This will automatically run on the test set (evaluation with extrapolation range) in the end of training.
To run only evaluation on a previously trained model use the extra flags `--test_mode` and `--use_ckpt`:

```
PYTHONPATH=. python runners/run_physics.py --task=spring_color --model=PhysicsNet --epochs=500 
--batch_size=100 --save_dir=<experiment_folder> --autoencoder_loss=3.0 --base_lr=3e-4 
--color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false 
--use_ckpt=true --test_mode=true 
```

This will use the checkpoint found in `<experiment_folder>`. To evaluate a checkpoint from a different folder use `--ckpt_dir`:

```
PYTHONPATH=. python runners/run_physics.py --task=spring_color --model=PhysicsNet --epochs=500 
--batch_size=100 --save_dir=<experiment_folder> --autoencoder_loss=3.0 --base_lr=3e-4 
--color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false 
--use_ckpt=true --test_mode=true --ckpt_dir=<folder_with_checkpoint>
```

To keep training a model from a checkpoint, simply use the same as above, but with `--test_mode=false`. Note that in this case `base_lr` will be used as the starting learning rate - there is no global learning rate variable saved in the checkpoint - so if you restart training after annealing was applied, be sure to change the `base_lr` accordingly.

Notes on flags, hyperparameters, and general training behavior:
* Using `--anneal_lr=true` will reduce the base learning rate by a factor of 5 after 70% of the epochs are completed. To change this find the corresponding code in `nn/network/base.py`, in the class method `BaseNet.train()`.
* When using `autoencoder_loss`, the encoder and decoder parts of the model will train fairly early in training. The rest of training is mostly improving the physical parameters, but this can take a long time. I recommend training between 500 and 1000 epochs (higher for `3bp_color` dataset, lower for `spring` datasets).


## Tasks

There are currently 5 tasks implemented in this repo: 

* `bouncing_balls`: (here there are no learnable physical parameters)
* `spring_color`: Two colored balls connected by a spring.
* `spring_color_half`: Same as above, but in the input and prediction range the balls never leave half of the image. They only move to the other half of the image in the extrapolation range of the test set.
* `mnist_spring_color`: Two colored MNIST digits connected by a spring, in a CIFAR background.
* `3bp_color`:  Three colored balls connected by gravitational force (`3bp` stands for 3-body-problem).
 
 The input, prediction and extrapolation steps are preset for each task, and correspond to the values described in the paper (see 1st paragraph of Section 4.1).
 
 ## Data
 
 The datasets for the tasks above can be downloaded from [this Google Drive](https://drive.google.com/open?id=16uvdhZiv2CkoDDDNGRG4l_T7LEZXzfyA). These datasets should be placed in a folder called `<repo_root>/data/datasets` in order to be automatically fetched by the code. 
 
 ## Hyperparameters
 
 For the tasks above, the recommended `base_lr` and `autoencoder_loss` paramters are:
 * `bouncing_balls`: `--base_lr=3e-4 --autoencoder_loss=2.0`
* `spring_color`: `--base_lr=6e-4 --autoencoder_loss=3.0`
* `spring_color_half`: `--base_lr=6e-4 --autoencoder_loss=3.0`
* `mnist_spring_color`: `--base_lr=6e-4 --autoencoder_loss=3.0`
* `3bp_color`:  `--base_lr=1e-3 --autoencoder_loss=5.0`
 
 ## Interpreting results in the `log.txt` file
 
 When tracking training progress from the `log.txt` file, a value of `eval_recons_loss` below 1.5 indicates that the encoder and decoder have correctly discovered the objects in the scene, and a value of `eval_pred_loss` below 3.0 and 30.0 (for balls and mnist datasets, respectively) indicates that the velocity estimator and the physical parameters have been learned correctly. Due to the dependency on initialization, it is possible that even using the hyperparameters above the model gets stuck in a local minimum and never gets below the aforementioned values, by failing to discover all the objects or learning the correct physical parameters/velocity estimator (this is common in unsupervised object discovery methods). I am working on improving convergence stability.
 
 ## Reading other results
 
 The `example%d.jpg` files show random rollouts from the validation/test set. The top row corresponds to the model prediction, middle row to the ground-truth, and bottom row to the reconstructed frames (as used by the autoencoder loss - this can be used to evaluate whether the objects have been discovered even though the dynamics might not have been learned yet).
 
 The `templates.jpg` file shows the learned contents (top) and masks (bottom). 
