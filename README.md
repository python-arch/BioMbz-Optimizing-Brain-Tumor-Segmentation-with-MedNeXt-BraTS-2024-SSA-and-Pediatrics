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
git clone [https://github.com/yourusername/BraTS-UGRIP.git](https://github.com/python-arch/BraTS-UGRIP.git/)

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

Instructions and examples for using the project. You can add screenshots to illustrate the features.

```bash
# Example command
your-command-here

# Example output
your-output-here
```

## Features

- Feature 1
- Feature 2
- Feature 3

## Contributing

Contributions are always welcome! Please follow these steps:

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Your Name - [your-email@example.com](mailto:your-email@example.com)

Project Link: [https://github.com/yourusername/your-repo-name](https://github.com/yourusername/your-repo-name)
