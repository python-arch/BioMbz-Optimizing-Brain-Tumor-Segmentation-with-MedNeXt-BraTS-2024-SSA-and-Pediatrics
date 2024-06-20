# Enhancing Brain Tumor Segmentation: Improving MedNext Model Architecture and Performance through Diverse Dataset Training

This project focuses on advancing the performance of the baseline MedNext model for brain tumor segmentation by exploring various methods to alter the model architecture and evaluating the impact of training with different datasets of varying sizes. Our work aims to provide robust solutions for segmenting brain tumors with higher accuracy and efficiency.

## Table of Contents
 
- [Installation](#installation)
- [Dataset Setup](#Dataset_Setup)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

Step-by-step instructions on how to get a development environment running.

```bash
# Clone the repository
git clone https://github.com/python-arch/BraTS-UGRIP.git

# Navigate to the project directory
cd BraTS-UGRIP

# Create Conda Environment
conda create BraTS-UGRIP python==3.8

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

## Features

- Feature 1
- Feature 2
- Feature 3

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
