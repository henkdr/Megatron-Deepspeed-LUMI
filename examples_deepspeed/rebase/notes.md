# Training GPT-3 with Megatron-Deepspeed on LUMI
 
## Setup instructions:
- Install PyTorch container:
    ```bash
    module load LUMI/23.03 partition/container EasyBuild-user
    eb PyTorch-2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240404.eb
    ```
- To install Megatron-Deepspeed in the container, follow the instructions in first code block [on this page](https://rocm.blogs.amd.com/artificial-intelligence/megatron-deepspeed-pretrain/README.html)
- Pack the installed packages in a SquashFS file using the instructions [here] (https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/p/PyTorch/#extending-the-containers-with-virtual-environment-support)
- Follow the rest of the instructions in the code block [here](https://rocm.blogs.amd.com/artificial-intelligence/megatron-deepspeed-pretrain/README.html) to download and pre-process the dataset.
- Change 'home_dir=/users/username;' in Megatron-DeepSpeed/examples_deepspeed/rebase/lumi_ds_pretrain_gpt.sh (line 34) to the directory above where your clone of the repository is located.

- Run the training job by submitting lumi_ds_pretrain_gpt.sh in /Megatron-DeepSpeed/examples_deepspeed/rebase/

## Notes:

- The launcher was changed from the deepspeed launcher to the torch launcher (torch.distributed.run) because the deepspeed launcher relies on password-less ssh to other nodes, which will not work on LUMI.
- The torch launcher expects 1 task per node.
- It is unclear how torchrun manages GPU assignment/binding internally.
	- Could numactl be used to control this?
    - See also https://github.com/pytorch/pytorch/issues/115305
- Training uses fp16 precision.
- In the deepspeed/megatron configuration in lumi_ds_pretrain_gpt_13b.sh, mp_size denotes the tensor-parallel degree.
- If the training job hangs on ‘compiling and loading fused kernels …’:
	- Remove megatron/fused_kernels/build
	- See also [this discussion](https://github.com/NVIDIA/Megatron-LM/issues/82)


## Training throughput results (small runs):

| Model Size | Nodes | PP* | TP* | DP* | Zero Stage | T** (samples/s) | T (FLOP/s per GPU (GCD)) | % of peak *** |
|------------|-------|-----|-----|-----|------------|-----------------|--------------------------|---------------|
| 125M       | 2     | 2   | 1   | 4   | 1          | 76.843          | 11.94                    | 6.23          |
| 125M       | 2     | 2   | 1   | 4   | 1          | 76.843          | 11.94                    | 6.23          |
| 13B        | 1     | 1   | 1   | 8   | 1          | 2.686           | 74.90                    | 39.11         |
| 13B        | 2     | 1   | 1   | 16  | 1          | 4.711           | 65.67                    | 34.29         |
| 13B        | 2     | 1   | 1   | 16  | 2          | 5.166           | 72.02                    | 37.61         |
| 13B        | 2     | 8   | 2   | 1   | 1          | 4.498           | 62.71                    | 32.75         |
| 13B        | 2     | 2   | 4   | 2   | 1          | ?               | ?                        | ?             |
| 13B        | 2     | 4   | 2   | 2   | 1          | ?               | ?                        | ?             |

\* PP = pipeline parallel degree, TP = tensor parallel degree, DP = data parallel degree

\** T = throughput

\*** Achieved FLOP/s as percentage of theoretical peak performance of one GCD

- Observed throughput is as expected, considering the results mentioned here:
	- [Scaling the pre-training of Large-Language Models of 100B parameters to thousands of amd MI250x GPU on LUMI](https://www.lumi-supercomputer.eu/scaling-the-pre-training-of-large-language-models-of-100b-parameters-to-thousands-of-amd-mi250x-gpus-on-lumi/)
		- Note that here, throughput is listed per MI250x, not per GCD.
    - [Results in this REAMDE](https://github.com/microsoft/Megatron-DeepSpeed/tree/3afd267e1e50b1410beb606c5625cc232a55417a/examples_deepspeed/rebase)
    - [Results in this README](https://github.com/microsoft/Megatron-DeepSpeed/tree/main)

## Running a different configuration
To run a different configuration (# nodes / GPUs, TP/PP/DP configuration), change the following parameters:

- In lumi_ds_pretrain_gpt.sh:
    - #SBATCH --nodes=4 (for example to 48)
- In lumi_ds_pretrain_gpt_13b.sh:
    - The model size (for example uncomment config GPT-3 175B)
	- mp_size=2 (for example to 8)
	- pp_size=2 (for example to 8)
	- num_gpus=16 (to #nodes * 8 GCDs)
	- num_node=2 (to #nodes)
