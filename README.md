# Enhancing Brain Tumor Segmentation: Improving MedNext Model Architecture and Performance through Diverse Dataset Training

This project focuses on advancing the performance of the baseline MedNext model for brain tumor segmentation by exploring various methods to alter the model architecture and evaluating the impact of training with different datasets of varying sizes. Our work aims to provide robust solutions for segmenting brain tumors with higher accuracy and efficiency.

## Table of Contents
 
- [Installation](#installation)
- [Dataset Setup](#Dataset_Setup)
- [Usage](#usage)
- [Features and Experiments OverView](#features)
- [Credits](#credits)

## Installation

Step-by-step instructions on how to get a development environment running.

```bash
# Clone the repository
git clone https://github.com/python-arch/BraTS-UGRIP.git

# Navigate to the project directory
cd BraTS-UGRIP

# Create Conda Environment
conda create -n BraTS-UGRIP python==3.8
# Activate the Conda Enviroment
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
- Exploring Attention-based MedneXt:
- Automating the process of training your model using `slurm`:

## Features and Experiments OverView
We conducted a total of more than 200 experiments, which included:
1. Integrating the new optimizer and Hyper-parameter tuning experiments.
2. Implementing Different Attention Mehcanisms in the MedNext.
3. Retraining the models for 75 epochs instead of the original 100 epochs.
4. Main experiments involving different training and evaluation setups:
    - Train on SSA, Evaluate on SSA.
    - Train on SSA + Glioma, Evaluate on SSA.
    - Train on SSA + Generated Data, Evaluate on SSA.
    - Train on SSA + Glioma + Generated Data, Evaluate on SSA.
5. Souping experiments to combine model checkpoints for improved performance.
6. Performaing Augmentations Experiments.

### "The Road Less Scheduled" - Why?
It is innovative momentum approach employs an alternative form of momentum with superior theoretical properties, ensuring worst-case optimal performance for any momentum parameter in convex Lipschitz settings. The authors conducted a comprehensive evaluation on 28 diverse problems, ranging from logistic regression to large-scale deep learning tasks, demonstrating that schedule-free methods often match or surpass heavily-tuned cosine schedules. Paper could be found here : https://arxiv.org/abs/2405.15682

### Hyper-Parameters Tuning: Integration of the New Optimizer into MedNext
1. We utilized the authors' experimental setup for working with an MRI image dataset as a starting point.
2. Conducted over 11 experiments on a specific data fold (fold number 2) over 100 epochs to identify the optimal hyper-parameters for the model, comparing the results against our baseline model.
3. Expanded the experiments to cover all 5 folds and performed 5-fold cross-validation to ensure the new model's performance consistency.

### Attention mechanisms experiments:
### Model Souping
#### Souping Approach
- **Uniform Souping:** Initially attempted this method but found it ineffective for our medical data.
- **Greedy Souping:** This approach proved more successful across different folds of our data:
    - **Fold 0:** Souped 15 checkpoints.
    - **Fold 1:** Souped 7 checkpoints.
    - **Fold 2:** Souped 6 checkpoints.

The souping process involved loading pre-trained models, sorting them by their test average scores, and iteratively adding models that improved the performance. Results showed a minor improvement of around 0.34% for the souped model using fold 0, with no significant improvements for other folds.

#### Automation of Training Process
**Challenges Addressed:**
- The necessity of running numerous models with various hyper-parameter combinations for souping posed significant logistical challenges.
- Manually managing and submitting each job was impractical and time-consuming, especially with the need for parallel processing capabilities.

**Solution:**
- Developed a Python script integrated with Slurm, a job scheduler for managing and submitting jobs to compute clusters.
- The script automates the process by checking available workstations for optimal resource allocation and dynamically submitting jobs to Slurm for efficient parallel processing of model training tasks.

This project demonstrates a comprehensive approach to improving brain tumor segmentation using the MedNext model, leveraging innovative optimization techniques, extensive hyper-parameter tuning, and efficient model souping methods.
### Augmentation Experiments
## Credits

This project was developed using MBZUAI Resources and as a part of the undergraduate research internship program. Thanks to all of our mentors and our professor for supporting us :)
