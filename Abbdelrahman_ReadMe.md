# "The Road Less Scheduled" - Why?
In our baseline MedNeXt models, we used traditional optimizers with learning rate schedulers. While these optimizers performed well in our experiments, a recent breakthrough by Meta AI introduced a novel optimizer with a unique setup. Let us present our new **Schedule-Free Optimizer**.

Traditional learning rate schedules often require specifying an optimization stopping point, T, to achieve optimal performance. In contrast, the Schedule-Free Optimizer offered by Meta AI eliminates the need for such schedules altogether. By avoiding explicit scheduling, this optimizer provides state-of-the-art performance across a wide range of problems, from convex optimization to large-scale deep learning tasks, without introducing additional hyperparameters beyond those used in standard momentum-based optimizers.

The novel Schedule-Free approach is grounded in a new theoretical framework that unifies scheduling and iterate averaging. This innovative momentum technique offers superior theoretical properties, ensuring optimal performance in convex Lipschitz settings for any momentum parameter. Comprehensive evaluations across 28 diverse problems, including logistic regression and large-scale deep learning tasks, show that this new approach often matches or surpasses the performance of heavily-tuned cosine schedules.

That's why this optimizer stands out. As we discuss our fine-tuning method, you'll see why we think this optimizer will be a key player for us.

For more details, you can access the full paper [here](https://arxiv.org/abs/2405.15682).


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

