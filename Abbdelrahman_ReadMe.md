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
The results demonstrated a slight improvement in the performance of the souped model using the models trained on Fold 0, with an increase of approximately 0.34%. However, for the other folds, no notable improvements were observed. The tables below provide a detailed comparison between the best model from each fold and the greedy souped model. In the following subsections we explain the metrics we used for our evaluation and the tables for the results.

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

</center>

