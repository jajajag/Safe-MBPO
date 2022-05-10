# Safe-MBPO
Code for the NeurIPS 2021 paper "Safe Reinforcement Learning by Imagining the Near Future" by Garrett Thomas, Yuping Luo, and Tengyu Ma.

Some code is slightly changed by Jag for experimental purpose. [Force](https://github.com/gwthomas/force) repo is merged into this repo.

## Installation
Required Python packages and environment variables can be setup simply running:

	sh install.sh

Once setup is complete, run the code using the following command:

	python main.py -c config/ENV.json

where ENV is replaced appropriately. To override a specific hyperparameter, add `-s PARAM VALUE` where `PARAM` is a string.
Use `.` to specify hierarchical structure in the config, e.g. `-s alg_cfg.horizon 10`.
