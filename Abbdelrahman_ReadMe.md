# Model Souping
1. **Uniform Souping:** Initially attempted this method but found it didn't perform well on our medical data. It didn't effectively enhance model performance as expected.

2. **Greedy Souping:** This approach proved more successful across different folds of our data:
    1. **Fold 0:** Souped 15 checkpoints.
    2. **Fold 1:** Souped 7 checkpoints.
    3. **Fold 2:** Souped 6 checkpoints.

3. The results showed improvement in the performance of the souped model using the models trained on fold 0. However, it was a minor improvement around 0.34%. For the other folds, no improvements were observed.

4. The following tables show a detailed comparison between the best model obtained in each fold and the greedy souped model.

<div style="display: flex; justify-content: center; align-items: center;">

<!-- First Column: Best Model Table -->
<div style="margin-right: 20px;">
<table>
<caption><strong>Performance of the Best Model (Fold 0)</strong></caption>
<thead>
<tr><th><strong>Metric</strong></th><th><strong>Value</strong></th></tr>
</thead>
<tbody>
<tr><td>test_avg</td><td>0.8220043778419495</td></tr>
<tr><td>test_et</td><td>0.7842099070549011</td></tr>
<tr><td>test_loss</td><td>0.19675968370089927</td></tr>
<tr><td>test_tc</td><td>0.8237403035163879</td></tr>
<tr><td>test_wt</td><td>0.8580629825592041</td></tr>
</tbody>
</table>
</div>

<!-- Second Column: Greedy Souping Model Table -->
<div>
<table>
<caption><strong>Performance of the Greedy Soup (Fold 0)</strong></caption>
<thead>
<tr><th><strong>Metric</strong></th><th><strong>Value</strong></th></tr>
</thead>
<tbody>
<tr><td>test_avg</td><td>0.8254003524780273</td></tr>
<tr><td>test_et</td><td>0.7878062725067139</td></tr>
<tr><td>test_loss</td><td>0.19331554230302572</td></tr>
<tr><td>test_tc</td><td>0.8270795941352844</td></tr>
<tr><td>test_wt</td><td>0.8613150715827942</td></tr>
</tbody>
</table>
</div>

</div>



| **Test Metric** | **DataLoader 0**      |
|-----------------|-----------------------|
| test_avg        | 0.8622822761535645     |
| test_et         | 0.8255845904350281     |
| test_loss       | 0.1538387987141808     |
| test_tc         | 0.8322740197181702     |
| test_wt         | 0.9289882779121399     |

**Table 3:** Performance of the best model (Fold 1) 6 checkpoints

| **Test Metric** | **DataLoader 0**      |
|-----------------|-----------------------|
| test_avg        | 0.8622822761535645     |
| test_et         | 0.8255845904350281     |
| test_loss       | 0.1538387987141808     |
| test_tc         | 0.8322740197181702     |
| test_wt         | 0.9289882779121399     |

**Table 4:** Performance of the greedy soup (Fold 1) 6 checkpoints

| **Test Metric** | **DataLoader 0**      |
|-----------------|-----------------------|
| test_avg        | 0.889476478099823      |
| test_et         | 0.8488584160804749     |
| test_loss       | 0.13734080673505863    |
| test_tc         | 0.8933649659156799     |
| test_wt         | 0.926206111907959      |

**Table 5:** Performance of the individual model (Fold 2) 7 checkpoints

| **Test Metric** | **DataLoader 0**      |
|-----------------|-----------------------|
| test_avg        | 0.889476478099823      |
| test_et         | 0.8488584160804749     |
| test_loss       | 0.13734080673505863    |
| test_tc         | 0.8933649659156799     |
| test_wt         | 0.926206111907959      |

**Table 6:** Performance of the greedy soup (Fold 2) 7 checkpoints

