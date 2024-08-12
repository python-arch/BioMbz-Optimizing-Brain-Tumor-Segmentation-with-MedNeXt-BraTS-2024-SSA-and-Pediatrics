# Optimizing Brain Tumor Segmentation with MedNeXt: BraTS 2024 SSA and Pediatrics

This project focuses on advancing the performance of the baseline MedNext model for brain tumor segmentation by exploring various methods to alter the model architecture and evaluating the impact of training with different datasets of varying sizes. Our work aims to provide robust solutions for segmenting brain tumors with higher accuracy and efficiency.

## Table of Contents
 
- [Installation](#installation)
- [Dataset Setup](#Dataset_Setup)
- [Usage](#usage)
- [Features and Experiments OverView](#features)

## Installation

Step-by-step instructions on how to get a development environment running.

```bash
# Clone the repository
git clone https://github.com/python-arch/BraTS-UGRIP.git

# Navigate to the project directory
cd BraTS-UGRIP

# Create Conda Environment
conda create -n BraTS-UGRIP python==3.8 -y

# Activate the enviroment

conda activate BraTS-UGRIP

# Install dependencies
pip install -r requirements.txt
```

## Dataset_Setup
- Make sure you have the folder for your data inside the `dataset` folder
- Make sure you have the `brats_ssa_2023_5_fold.json` with the data and folds filled.
- Make sure you replace the paths for the dataset folder , the preprocessed dataset folder , and the json file in their corresponding fields in the code.
- If you want to integrate new data in the dataset (aka. Generated Data or Glioma Data) you can use these scripts:
    - `preprocess_generated_data.py` to preprocess the generated data folder and integrate it with the existing data.
    - Similarly , `sample_glioma_dataset.py` can be used to sample the glioma dataset and integrate it with the existing training data.
- It should be noted that for preprocessing the generated data and glioma data scripts , these data is always appended to fold -1 (which means it is used only for training).
- All Evaluations will be done on the original (aka. SSA Dataset) which is splitted into 5 folds.

## Usage
- Training your own MedNext model:
   - After setting up the dataset , you can utilize `mednext_train.py`:
      - You will need to have account on wandb to log the results , and setting up the `Args` Class to setup the hyper-parameters for the experiment.
      - The difference between the `Args` in this training file and the original `mednext_train.py` is that we have the option for new `schedule-free` optimizer.
- Souping:
   - You can use the `souping.py` file in which you will have to provide the path for the folder containing pre-trained models.
   - Moreover, you need to specify in the Args class the value for the `greedy` attribute if it is **False** Uniform Souping will be performed , otherwise , Greedy Souping.

# Features and Experiments OverView
# "The Road Less Scheduled" - Why?
In our baseline MedNeXt models, we used traditional optimizers with learning rate schedulers. While these optimizers performed well in our experiments, a recent breakthrough by Meta AI introduced a novel optimizer with a unique setup. Let us present our new **Schedule-Free Optimizer**.

Traditional learning rate schedules often require specifying an optimization stopping point, T, to achieve optimal performance. In contrast, the Schedule-Free Optimizer offered by Meta AI eliminates the need for such schedules altogether. By avoiding explicit scheduling, this optimizer provides state-of-the-art performance across a wide range of problems, from convex optimization to large-scale deep learning tasks, without introducing additional hyperparameters beyond those used in standard momentum-based optimizers.

The novel Schedule-Free approach is grounded in a new theoretical framework that unifies scheduling and iterate averaging. This innovative momentum technique offers superior theoretical properties, ensuring optimal performance in convex Lipschitz settings for any momentum parameter. Comprehensive evaluations across 28 diverse problems, including logistic regression and large-scale deep learning tasks, show that this new approach often matches or surpasses the performance of heavily-tuned cosine schedules.

That's why this optimizer stands out. As we discuss our fine-tuning method, you'll see why we think this optimizer will be a key player for us.

For more details, you can access the full paper [here](https://arxiv.org/abs/2405.15682).

# Methodology (How did we tune our experimental setup for this optimizer)
To tune our experimental setup using our new optimizer we have done several experiments with several folds and several learning rates to find our optimal set of hyper-parameters. The found optimal parameters were learning rate of 0.0027 and weight decay of zero. In the figure below , you can see the validation loss curves for different experiments we have carried to tune the our model using the new optimizer.

![W B Chart 8_12_2024, 1_22_51 AM](https://github.com/user-attachments/assets/d82e4f12-6722-46f5-89cb-c2521952dec0)

# Model Souping
### What is Model Souping?
Model souping is a concept that was introduced back in 2022 , aiming to boost ML/DL models performance and their generalization ability. Typically, to get the best model accuracy, you train several models with different settings and then choose the one that performs the best on a separate validation set, while ignoring the others. Instead of picking just one model, averaging the weights of several models trained with different settings or hyper-parameters can actually boost accuracy and make the model more reliable. The best part is, this method doesn’t add extra costs for running or storing the models like , for example , ensemble technique. This introduced recipe is called "Model Soups".

### Types of Model Souping

There are two main types of model souping: **Uniform Souping** and **Greedy Souping**.

#### Uniform Souping

In uniform souping, the weights of all selected models are averaged equally. This method assumes that each model contributes equally to the final performance. However, in practice, uniform souping may not always lead to the best results, especially when the performance of the individual models varies significantly.

#### Greedy Souping

Greedy souping, on the other hand, involves a more selective process. Instead of averaging all models, this method adds models to the soup one by one, selecting only those that contribute positively to the overall performance. Greedy souping can result in better performance as it focuses on combining only the best-performing models or those that complement each other well.

#### Our Methodolgy
We experimented with both uniform souping and greedy souping techniques on our models while implementing a 5-fold cross-validation process. To assess the impact of model souping on performance, we trained models across different folds and applied souping techniques to each fold. We then compared the performance of the best model from each fold with that of the souped model. In the following subsections we provide details for our methodology on both uniform and greedy souping:

1. **Uniform Souping:** Initially attempted this method but found it didn't perform well on our medical data. It didn't effectively enhance model performance as expected.

2. **Greedy Souping:** This approach proved more successful across different folds of our data:
    1. **Fold 0:** Souped 15 checkpoints.
    2. **Fold 1:** Souped 7 checkpoints.
    3. **Fold 2:** Souped 6 checkpoints.
  
#### Results
The results demonstrated a slight improvement in the performance of the souped model using the models trained on Fold 0, with an increase of approximately 0.34%. However, for the other folds, no notable improvements were observed. The tables below provide a detailed comparison between the best model from each fold and the greedy souped model. Moreover , the used metrics in the comparison are explained below.

#### Metrics Used

- **test_avg**: Average Dice score of the model.
- **test_et**: Dice score for the enhancing tumor (ET).
- **test_tc**: Dice score for the tumor core (TC).
- **test_wt**: Dice score for the whole tumor (WT).
- **test_loss**: Loss value of the model.

<center>

| **Metric**   | **Best Model (Fold 0)** | **Greedy Soup (Fold 0)** |
|--------------|-------------------------|--------------------------|
| **test_avg** | 0.8220043778419495       | 0.8254003524780273        |
| **test_et**  | 0.7842099070549011       | 0.7878062725067139        |
| **test_loss**| 0.19675968370089927      | 0.19331554230302572       |
| **test_tc**  | 0.8237403035163879       | 0.8270795941352844        |
| **test_wt**  | 0.8580629825592041       | 0.8613150715827942        |

**Table 1:** Performance comparison for Fold 0

</center>

<center>

| **Metric**   | **Best Model (Fold 1)** | **Greedy Soup (Fold 1)** |
|--------------|-------------------------|--------------------------|
| **test_avg** | 0.8622822761535645       | 0.8622822761535645        |
| **test_et**  | 0.8255845904350281       | 0.8255845904350281        |
| **test_loss**| 0.1538387987141808       | 0.1538387987141808        |
| **test_tc**  | 0.8322740197181702       | 0.8322740197181702        |
| **test_wt**  | 0.9289882779121399       | 0.9289882779121399        |

**Table 2:** Performance comparison for Fold 1

</center>

<center>

| **Metric**   | **Best Model (Fold 2)** | **Greedy Soup (Fold 2)** |
|--------------|-------------------------|--------------------------|
| **test_avg** | 0.889476478099823        | 0.889476478099823         |
| **test_et**  | 0.8488584160804749       | 0.8488584160804749        |
| **test_loss**| 0.13734080673505863      | 0.13734080673505863       |
| **test_tc**  | 0.8933649659156799       | 0.8933649659156799        |
| **test_wt**  | 0.926206111907959        | 0.926206111907959         |

**Table 3:** Performance comparison for Fold 2

# Patch size
for both brats Africa and BraTS Pediatric we have used a patch size of (128,160,112) instead of (128 , 128 , 128 ) , we can see performance boost in ET and WT from this table when we change the patch size 

| Metric     | Patch Size (128, 128, 128) | Patch Size (128, 160, 112) |
|------------|----------------------------|----------------------------|
| Dice ET    | 0.821                      | 0.867                      |
| Dice TC    | 0.811                      | 0.869                      |
| Dice WT    | 0.891                      | 0.932                      |
| HD95 ET    | 37.010                     | 15.578                     |
| HD95 TC    | 43.803                     | 22.145                     |
| HD95 WT    | 24.908                     | 8.833                      |

*ET: Enhancing Tumor, TC: Tumor Core, WT: Whole Tumor, HD95: 95th percentile Hausdorff Distance*

# Synthetic Data Generation for BraTS Africa

## Overview

We are leveraging the [Med-DDPM repository](https://github.com/JotatD/med-ddpm-brats) to generate high-quality synthetic data for brain MRI images. This project extends the original work by fine-tuning the model on the BraTS Africa dataset using LoRA (Low-Rank Adaptation) techniques.

## Med-DDPM: Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis

Med-DDPM is a powerful tool for generating realistic and semantically accurate 3D brain MRI images. It supports both whole-head MRI synthesis and brain-extracted 4 modalities MRIs (T1, T1ce, T2, Flair) based on the BraTS2021 dataset.

### Key Features of Med-DDPM

- Generates high-quality 3D medical images while preserving semantic information
- Trained on whole-head MRI and brain-extracted 4 modalities MRIs
- Provides pretrained model weights for immediate use
- Supports custom dataset integration

### Visual Examples

The following images demonstrate the capability of Med-DDPM in generating synthetic brain MRI images:

<table>
  <tr>
    <td align="center">
      <strong>Input Mask</strong><br>
      <img src="images/img_0.gif" alt="Input Mask" width="100%">
    </td>
    <td align="center">
      <strong>Real Image</strong><br>
      <img src="images/img_1.gif" alt="Real Image" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Synthetic Sample 1</strong><br>
      <img src="images/img_2.gif" alt="Synthetic Sample 1" width="100%">
    </td>
    <td align="center">
      <strong>Synthetic Sample 2</strong><br>
      <img src="images/img_3.gif" alt="Synthetic Sample 2" width="100%">
    </td>
  </tr>
</table>

These gifs showcase:
1. The input segmentation mask
2. A real brain MRI image
3. Two different synthetic samples generated from the same input mask

Note how the synthetic samples maintain the overall structure defined by the input mask while introducing realistic variations in tissue appearance and intensity.

## Our Approach: LoRA Fine-tuning

To adapt Med-DDPM for our specific use case with the BraTS Africa dataset, we implemented LoRA fine-tuning on the last two layers of the model. This approach allows us to:

1. Efficiently adapt the pretrained model to our domain-specific data
2. Maintain the core capabilities of the original model while introducing dataset-specific features
3. Reduce computational resources required for fine-tuning

## Synthetic Data Generation Process

1. **Base Model**: We start with the pretrained Med-DDPM model.
2. **Fine-tuning**: Apply LoRA fine-tuning on the last two layers using the BraTS Africa dataset.
3. **Generation**: Use the fine-tuned model to generate synthetic brain MRI data.
4. **Integration**: Incorporate the synthetic data into our training pipeline.

## Performance Evaluation

We have conducted extensive evaluations to assess the quality and utility of our synthetic data. The performance metrics and comparative analyses can be found in the "Performance" section below.

## Usage Instructions

1. Clone the Med-DDPM repository:
   ```
   git clone https://github.com/JotatD/med-ddpm-brats.git
   cd med-ddpm-brats
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download our LoRA weights (link to be provided).

4. Generate synthetic samples:
   ```
   ./scripts/sample_brats_lora.sh
   ```

## Performance

| Metric     | BRATS Synthetic | BRATS Base |
|------------|-----------------|------------|
| Dice ET    | 0.876           | 0.867      |
| Dice TC    | 0.866           | 0.869      |
| Dice WT    | 0.927           | 0.932      |
| HD95 ET    | 15.328          | 15.578     |
| HD95 TC    | 22.078          | 22.145     |
| HD95 WT    | 8.960           | 8.833      |

*ET: Enhancing Tumor, TC: Tumor Core, WT: Whole Tumor, HD95: 95th percentile Hausdorff Distance*


# Model Ensembling Algorithm

## Description

This algorithm performs model ensembling for brain MRI scans, combining predictions from multiple models with weighted averaging.

## Input

- `N` models, each with corresponding weightings for every channel (TC, WT, ET)
- An input brain MRI scan `x ∈ ℝ^(3×H×W×D)`

## Algorithm

1. Initialize output `y` as a zero tensor with shape `3 × H × W × D`
2. Initialize `sum_w` as a zero vector with shape `3`
3. For each model `n` from 1 to N:
   - Add the weighted prediction of model `n` to `y`:
     `y = y + models[n](x) * weightings[n]`
   - Add the weighting of model `n` to `sum_w`:
     `sum_w = sum_w + weightings[n]`
4. Normalize the output:
   `y = y / sum_w`
5. Return `y`

## Output

The final ensemble prediction `y` for the input brain MRI scan.

## Notes

- The algorithm assumes that all models have the same input and output shapes.
- Weightings are applied per channel (TC, WT, ET).
- The final normalization ensures that the ensemble prediction is a weighted average of individual model predictions.


# MedNeXt Fine-tuning for BraTS Africa Dataset

This part focuses on fine-tuning the MedNeXt model, a state-of-the-art architecture for medical image segmentation, specifically for the BraTS Africa dataset. The base MedNeXt model was initially trained on a combined dataset of BraTS Adult Glioma and BraTS Africa. Our fine-tuning process aims to enhance the model's performance and specificity for brain tumor segmentation in African populations, addressing potential regional variations in tumor characteristics and imaging protocols.

## Algorithm

1. Model Initialization:
   - Load pre-trained MedNeXt model (trained on BraTS Adult Glioma + BraTS Africa)
   - Load weights from checkpoint: `args.loading_checkpoint`

2. Dataset Preparation:
   - Load BraTS Africa dataset using `json_brats2021_fold`
   - Create `DataModule` with `get_train_val_dataset()` function
   - Set ROI size: (128, 160, 112)

3. Hyperparameter Configuration:
   - Learning rate: 0.0027
   - Batch size: 2
   - Maximum epochs: 150
   - Optimizer: AdamW with ScheduleFree
   - Weight decay: 0
   - Loss function: Defined by `get_loss_fn(args.loss_type, args.mean_batch)`

4. Model Architecture Modification:
   - Freeze all layers except the last 26 layers
   - If `args.reset_weights` is True:
     - Iterate through last 10 layers of `model_layers`
     - Reset parameters of convolutional layers using `reset_parameters()`

5. Training Setup:
   - Initialize PyTorch Lightning `Trainer` with:
     - GPUs: `args.n_gpus`
     - Max epochs: `args.max_epochs`
     - Gradient checkpointing: Outside block style
     - Precision: 32-bit
     - Callbacks: `checkpoint_callback`, `lr_monitor`
   - Initialize WandB logger with project name: 'finetune-brats2023-adult-glioma'

6. Fine-tuning Process:
   - Execute `trainer.fit(module, dm)`
   - Use `PolynomialLR` scheduler if `args.lr_scheduler == 'polynomial'`
   - Apply deep supervision if `args.deep_sup == True`

7. Inference and Evaluation:
   - Perform sliding window inference with:
     - ROI size: (args.roi_x, args.roi_y, args.roi_z)
     - Overlap: args.infer_overlap
     - SW batch size: args.sw_batch_size
   - Apply sigmoid activation to output
   - Calculate metrics:
     - Dice score for ET (Enhancing Tumor), TC (Tumor Core), WT (Whole Tumor)
     - 95% Hausdorff Distance (HD 95) for ET, TC, WT

8. Post-processing:
   - If not `args.all_samples_as_train`:
     - Run `trainer.test(module, dataloaders=dm.test_dataloader(), ckpt_path='best')`

9. Result Logging:
   - Log metrics to WandB
   - Save best model based on validation performance

## Results

| S.No | Model Name | Dice Score ||| Dice Avg | HD 95 ||| HD 95 Avg |
|------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| | | ET | TC | WT | | ET | TC | WT | |
| 1 | MedNeXt Base | 0.867 | 0.869 | 0.932 | 0.889 | 15.578 | 22.145 | 8.833 | 15.519 |
| 2 | MedNeXt Medium | 0.875 | 0.850 | 0.933 | 0.886 | 14.472 | 31.639 | 8.349 | 18.820 |
| 3 | MedNeXt Ensemble (Base + Medium) | 0.852 | 0.839 | 0.912 | 0.868 | 12.406 | 26.090 | 10.431 | 16.309 |
| 4 | MedNeXt Finetuned True 0.5 | 0.876 | 0.870 | 0.933 | 0.893 | 15.328 | 22.040 | 8.746 | 15.371 |
| 5 | MedNeXt Finetuned True 0.7 | 0.883 | 0.873 | 0.926 | 0.894 | 14.248 | 21.028 | 9.017 | 14.764 |
| 6 | MedNeXt Finetuned False 0.5 | 0.874 | 0.870 | 0.933 | 0.892 | 15.320 | 22.039 | 8.805 | 15.388 |
| 7 | MedNeXt Finetuned False 0.7 | 0.882 | 0.874 | 0.928 | 0.895 | 14.269 | 21.097 | 8.994 | 14.787 |
| 8 | MedNeXt Finetuned True 0.7,0.5 | 0.883 | 0.873 | 0.933 | **0.896** | 14.248 | 21.028 | 8.770 | **14.682** |

Notes:
- ET: Enhancing Tumor
- TC: Tumor Core
- WT: Whole Tumor
- Bold values indicate best performance for that metric

The results show that the fine-tuned MedNeXt models generally outperform the base and medium models, with the "MedNeXt Finetuned True 0.7,0.5" configuration achieving the best overall performance in terms of average Dice score (0.896) and average HD 95 (14.682).

# Too many experiments to carry? Let's Automate it
## Automation Script for Model Training
To streamline and accelerate our model training process, we developed an automation script designed to handle the training of multiple models efficiently on faster GPU devices. This script facilitates the generation of over 100 models, which are essential for tasks such as souping, augmentation, and hyperparameter optimization.
### Key Features

- **Parameter Management:** The script uses a `ParameterManager` class to define and manage various hyperparameters for training. It generates all possible combinations of these parameters and queues them for execution.
  
- **SLURM Integration:** It automatically generates and submits SLURM job scripts for each parameter combination. These scripts are designed to run on GPU nodes, and the script handles job scheduling, execution, and monitoring.

- **Status Monitoring:** The script tracks the status of each job by monitoring output files generated by SLURM. It updates the status in real-time, including job start times, training progress, and completion.

- **Error Handling:** It includes mechanisms for handling errors, such as retrying failed jobs or jobs that exceed a timeout limit.

- **Model Saving:** Once training is completed, the script automatically copies the saved models to a designated directory for further analysis.

### How It Works

1. **Parameter Generation:** The script initializes with a set of hyperparameters and generates all possible combinations. These combinations are then queued for processing.

2. **Job Submission:** For each parameter combination, a SLURM job script is generated and submitted to a GPU node. The script handles job scheduling and execution.

3. **Monitoring and Updates:** The script continuously monitors job status by checking output files. It updates the training status, manages job failures, and handles timeouts.

4. **Completion and Saving:** Upon job completion, the script copies the resulting model checkpoints to a specified directory and updates the status.

This automation script significantly reduces the manual effort required for model training and ensures that a large number of models can be efficiently trained and evaluated. The figure below illustrates the results of one of our experiments. It displays the parameters for each model run, their current status, the GPU device they are using, the SLURM job number, and the epoch number.

<img width="824" alt="Screenshot 2024-08-11 at 12 28 50 AM" src="https://github.com/user-attachments/assets/d4608ee8-d0fc-4bd8-8ef6-23c5a49f062c">

</center>

# Brats2024 MLCube Submission

## Contents
- [Setup](#setup)
- [Installation](#installation)
- [Create MLCube](#create-mlcube)
- [Run MLCube](#run-mlcube)
- [Compatibility Test](#check-container-compatibility-locally)
- [Submit MLCube](#mlcube-submission)




## Setup

- Create/Activate your conda environment [Optional]

```
conda create -n medperf-env python=3.9 && \
conda activate medperf-env

```

## Installation

- Visit https://docs.medperf.org/mlcubes/mlcube_models/
- Install MedPerf

``` 
git clone https://github.com/mlcommons/medperf.git && \
cd medperf && \
pip install -e ./cli && \
medperf --version

```

- Make sure you have Docker installed 
- Install docker if not installed: Follow instructions in https://docs.docker.com/get-docker/

```
docker --version
```


## Create MLCube

- create medperf template using 
``` 
medperf mlcube create model

```

You will see the directory tree:
```
.
└── model_mlcube
    ├── mlcube
    │   ├── mlcube.yaml
    │   └── workspace
    │       └── parameters.yaml
    └── project
        ├── Dockerfile
        ├── mlcube.py
        └── requirements.txt

```

- Put your code inside mlcube/project folder

- change dir to mlcube folder
```
cd project_name/mlcube
```

- Build MLCube: Run this command inside mlcube folder
```
mlcube configure -Pdocker.build_strategy=always
```

## Run MLCube

- Run MLCube: Run this command inside mlcube folder
```
mlcube run --task infer data_path=<path to validation/test dataset> output_path=<dir for saving output> --gpus <number of gpus>
```

## Check Container Compatibility Locally

<!-- - Visit https://www.synapse.org/#!Synapse:syn51156910/wiki/622674

- login to medperf through synapse

```

- medperf auth synapse_login

``` -->

- First push your docker image to ghrc and rename your docker image in mlcube.yaml accordingly
- Download and unzip the dummy dataset files: https://storage.googleapis.com/medperf-storage/BraTS2023/test_mlcubes.tar.gz

- Run the following command to test your mlcube on the dummy dataset

```
medperf --gpus 1 test run \
   --demo_dataset_url synapse:syn52276402 \
   --demo_dataset_hash "16526543134396b0c8fd0f0428be7c96f2142a66" \
   -p <path to prep_segmentation folder> \
   -m ./mlcube \
   -e <path to eval_segmentation folder>  \
   --offline --no-cache
```


## MLCube Submission

### Docker Container

- Log into the Synapse Docker registry with your Synapse credentials.

```
docker login docker.synapse.org --username <syn_username>
```

- Rename the created image

```
docker tag <curr image name> docker.synapse.org/<project_synID>/<new image name>:<tag>
```
- Push docker container to synapse

```
docker push docker.synapse.org/<synID>/<image name>:<tag>
```


### MLCube Config Files

- Update the docker image name in mlcube.yaml file

- Follow the instructions here to create a .tar.gz file: https://docs.medperf.org/concepts/mlcube_files/

- Upload the .tar.gz file to synapse project 
- Submit the Docker using "Submit Docker Repository to Challenge" and "Submit File to Challenge".
- Make sure to use same "Submission name" for .tar.gz and docker container.

## Acknowledgements

We extend our gratitude to the original Med-DDPM authors and the BraTS Africa dataset creators. This work builds upon their valuable contributions to the field of medical image synthesis and analysis.

## Citation

If you use this adapted model or the synthetic data in your research, please cite both the original Med-DDPM paper and our work (details to be added upon publication).

<!-- # ensembling

so we training the dataset on 5 folds then we are ensembling all the 5 weights using the algorithm below -->
