# Chest X-ray Anomaly Detection with NVAE

## About NVAE
["NVAE: A Deep Hierarchical Variational Autoencoder"](https://arxiv.org/abs/2007.03898) is a deep hierarchical variational autoencoder that enables training SOTA 
likelihood-based generative models on several image datasets.

## NVAE for anomaly detection
As original code consist of training/evaluating NVAE which is the generative model, I come up with applying NVAE to anomaly detection. The original implementation contains code for various datasets such as MNIST, CIFAR10, Celeb-A, and etc. However, I add extra dataset named Chest Radiography Dataset. You can access to this dataset from [here](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).

The Chest Radiography Dataset consist of a total of 4 classes: Normal, Covid, Lung-Opacity, and Viral Pneumonia. In order to perform anomaly detection, training and evaluation are conducted using only Normal and Covid data among them. Normal data is used for training, and Covid data is used for evaluation.


<p align="center">
    <img src="/home/ys/repo/NVAE/covid_dataset/COVID/COVID-89.png" width="200" /> 
    <img src="/home/ys/repo/NVAE/covid_dataset/Normal/Normal-79.png" width="200">
    
</p>
<p align="center">(left) Covid Image | (right) Normal Image <p align="center">

<br>
</br>

## Below we describe the code guidelines for training NVAE models.
---
## Requirements
NVAE is built in Python 3.7 using PyTorch 1.6.0. Use the following command to install the requirements:
```
pip install -r requirements.txt
``` 

## Running the main NVAE training and evaluation scripts
We use the following commands on each dataset for training NVAEs on each dataset for 
Table 1 in the [paper](https://arxiv.org/pdf/2007.03898.pdf). In all the datasets but MNIST
normalizing flows are enabled. Check Table 6 in the paper for more information on training
details. Note that for the multinode training (more than 8-GPU experiments), we use the `mpirun` 
command to run the training scripts on multiple nodes. Please adjust the commands below according to your setup. 
Below `IP_ADDR` is the IP address of the machine that will host the process with rank 0 
(see [here](https://pytorch.org/tutorials/intermediate/dist_tuto.html#initialization-methods)). 
`NODE_RANK` is the index of each node among all the nodes that are running the job.

<details><summary>Covid</summary>

GPU are used for training NVAE on dynamically binarized Covid.

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
cd $CODE_DIR
python train.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID --dataset covid --batch_size 200 \
        --epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 2 --use_se --res_dist --fast_adamax 
```
</details>

**If for any reason your training is stopped, use the exact same commend with the addition of `--cont_training`
to continue training from the last saved checkpoint. If you observe NaN, continuing the training using this flag
usually will not fix the NaN issue.**


## Monitoring the training progress
While running any of the commands above, you can monitor the training progress using Tensorboard:

<details><summary>Click here</summary>

```shell script
tensorboard --logdir $CHECKPOINT_DIR/eval-$EXPR_ID/
```
Above, `$CHECKPOINT_DIR` and `$EXPR_ID` are the same variables used for running the main training script.

</details> 

## Post-training sampling, evaluation, and checkpoints

<details><summary>Evaluating Log-Likelihood</summary>

You can use the following command to load a trained model and evaluate it on the test datasets:

```shell script
cd $CODE_DIR
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR/mnist --eval_mode=evaluate --num_iw_samples=1000
```
Above, `--num_iw_samples` indicates the number of importance weighted samples used in evaluation. 
`$CHECKPOINT_DIR` and `$EXPR_ID` are the same variables used for running the main training script.
Set `--data` to the same argument that was used when training NVAE (our example is for MNIST).

</details> 

<details><summary>Sampling</summary>

You can also use the following command to generate samples from a trained model:

```shell script
cd $CODE_DIR
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --eval_mode=sample --temp=0.6 --readjust_bn
```
where `--temp` sets the temperature used for sampling and `--readjust_bn` enables readjustment of the BN statistics
as described in the paper. If you remove `--readjust_bn`, the sampling will proceed with BN layer in the eval mode 
(i.e., BN layers will use running mean and variances extracted during training).

</details>

<details><summary>Computing FID</summary>

You can compute the FID score using 50K samples. To do so, you will need to create
a mean and covariance statistics file on the training data using a command like:

```shell script
cd $CODE_DIR
python scripts/precompute_fid_statistics.py --data $DATA_DIR/cifar10 --dataset cifar10 --fid_dir /tmp/fid-stats/
```
The command above computes the references statistics on the CIFAR-10 dataset and stores them in the `--fid_dir` durectory.
Given the reference statistics file, we can run the following command to compute the FID score:

```shell script
cd $CODE_DIR
python evaluate.py --checkpoint $CHECKPOINT_DIR/eval-$EXPR_ID/checkpoint.pt --data $DATA_DIR/cifar10 --eval_mode=evaluate_fid  --fid_dir /tmp/fid-stats/ --temp=0.6 --readjust_bn
```
where `--temp` sets the temperature used for sampling and `--readjust_bn` enables readjustment of the BN statistics
as described in the paper. If you remove `--readjust_bn`, the sampling will proceed with BN layer in the eval mode 
(i.e., BN layers will use running mean and variances extracted during training).
Above, `$CHECKPOINT_DIR` and `$EXPR_ID` are the same variables used for running the main training script.
Set `--data` to the same argument that was used when training NVAE (our example is for MNIST).

</details> 


## How to construct smaller NVAE models
In the commands above, we are constructing big NVAE models that require several days of training
in most cases. If you'd like to construct smaller NVAEs, you can use these tricks:

* Reduce the network width: `--num_channels_enc` and `--num_channels_dec` are controlling the number
of initial channels in the bottom-up and top-down networks respectively. Recall that we halve the
number of channels with every spatial downsampling layer in the bottom-up network, and we double the number of
channels with every upsampling layer in the top-down network. By reducing
`--num_channels_enc` and `--num_channels_dec`, you can reduce the overall width of the networks.

* Reduce the number of residual cells in the hierarchy: `--num_cell_per_cond_enc` and 
`--num_cell_per_cond_dec` control the number of residual cells used between every latent variable
group in the bottom-up and top-down networks respectively. In most of our experiments, we are using
two cells per group for both networks. You can reduce the number of residual cells to one to make the model
smaller.

* Reduce the number of epochs: You can reduce the training time by reducing `--epochs`.

* Reduce the number of groups: You can make NVAE smaller by using a smaller number of latent variable groups. 
We use two schemes for setting the number of groups:
    1. An equal number of groups: This is set by `--num_groups_per_scale` which indicates the number of groups 
    in each scale of latent variables. Reduce this number to have a small NVAE.
    
    2. An adaptive number of groups: This is enabled by `--ada_groups`. In this case, the highest
    resolution of latent variables will have `--num_groups_per_scale` groups and 
    the smaller scales will get half the number of groups successively (see groups_per_scale in utils.py).
    We don't let the number of groups go below `--min_groups_per_scale`. You can reduce
    the total number of groups by reducing `--num_groups_per_scale` and `--min_groups_per_scale`
    when `--ada_groups` is enabled.



## License
Please check the LICENSE file. NVAE may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please contact 
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).

You should take into consideration that VAEs are trained to mimic the training data distribution, and, any 
bias introduced in data collection will make VAEs generate samples with a similar bias. Additional bias could be 
introduced during model design, training, or when VAEs are sampled using small temperatures. Bias correction in 
generative learning is an active area of research, and we recommend interested readers to check this area before 
building applications using NVAE.

## Bibtex:
Please cite our paper, if you happen to use this codebase:

```
@inproceedings{vahdat2020NVAE,
  title={{NVAE}: A Deep Hierarchical Variational Autoencoder},
  author={Vahdat, Arash and Kautz, Jan},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
