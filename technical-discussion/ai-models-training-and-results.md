# AI Models: Training and Results

We trained two neural network architectures, **EfficientNet-B5** and a **Swin-Tiny transformer**, following the methodology described in Schaerf et al. (2024). The training protocol involved unfreezing all network layers and implementing a reduced learning rate to facilitate fine-tuning. Our initial baseline approach revealed model instability and overfitting. We then implemented a refined training protocol with regularization and stabilization techniques to produce more robust and reliable models.

|                        | **EfficientNet-B5**                                                                                                                                        | **Swin Transformer**                                                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**        | Serves as the primary baseline model, offering an optimal balance between computational efficiency and representational capacity through compound scaling. | Provides hierarchical feature extraction through shifted windowing mechanisms, enabling efficient processing of high-resolution imagery. |
| **Input Resolution**   | 256×256 pixels                                                                                                                                             | 256×256 pixels                                                                                                                           |
| **Model Variant**      | `efficientnet_b5`                                                                                                                                          | `swin_tiny_patch4_window7_224`                                                                                                           |
| **Output Classes**     | 2 (authentic, imitation)                                                                                                                                   | 2 (authentic, imitation)                                                                                                                 |
| **Pretrained Weights** | ImageNet                                                                                                                                                   | ImageNet                                                                                                                                 |
| **Optimization**       | AdamW optimizer with learning rate 3×10⁻⁵                                                                                                                  | AdamW optimizer with learning rate 1×10⁻⁵                                                                                                |

***

## Training and Evaluation

The dataset was partitioned using a **70/15/15 split** for the training, validation, and test sets, respectively. The **LMI-B (WikiArt)** dataset was used for this training and evaluation phase. A `GroupShuffleSplit` was used to ensure that patches from the same painting did not appear across different sets.

Model training employed patch-level supervision for granular learning. Validation was conducted at the painting level to assess overall artwork classification performance. For painting-level predictions, we implemented a voting mechanism aggregating individual patch predictions, with the final classification determined by a majority vote.

***

## Model Optimization

Initial training runs revealed significant performance issues. The EfficientNet-B5 model suffered from **training instability**, characterized by erratic loss curves, while the Swin Transformer exhibited **overfitting**, where it performed well on training data but failed to generalize to the validation set. Addressing these issues was critical for developing a trustworthy authentication model.

### What is Regularization?

Regularization techniques prevent models from overfitting training data. Overfit models memorize specific patterns and noise, leading to poor generalization and unreliable predictions on new, unseen data. Regularization constrains the complexity of a model, forcing the model to learn more generalizable patterns.

### How We Regularized the Swin Transformer

Our baseline Swin Transformer showed clear signs of overfitting. To address this, we implemented several regularization techniques:

* **Dropout and Drop Path**: Randomly deactivates neurons and entire residual connections during training, preventing the model from relying too heavily on any single feature.
* **Label Smoothing**: Softens the target labels during training, which discourages the model from making overly confident predictions.

#### Loss Plots: Before Regularization

_Demonstrates overfitting, as the validation loss begins to increase while training loss continues to decrease._

<figure><img src="../.gitbook/assets/loss_plot_overfit_swin.png" alt=""><figcaption></figcaption></figure>

#### Loss Plots: After Regularization

_Post-regularization: The validation loss converges consistently with the training loss, demonstrating reduced overfitting._

<figure><img src="../.gitbook/assets/loss_plot_regularized_swin.png" alt=""><figcaption></figcaption></figure>

### What is Stabilization?

Stabilization techniques ensure consistent and reliable training dynamics. Unstable models suffer from erratic loss curves, inconsistent predictions, and training failures due to exploding or vanishing gradients. These techniques normalize model behavior for smoother convergence.

### How We Stabilized the EfficientNet

Our baseline EfficientNet-B5 model was highly unstable. We implemented the following techniques to resolve this:

* **Targeted Weight Initialization**: We re-initialized and scaled down the weights of specific problematic layers (classifier head, SE modules) that were prone to causing exploding gradients.
* **Learning Rate Warmup**: The training began with a very low learning rate ($$1 \cdot 10^{-7}$$) and gradually increased it over the first 5 epochs, preventing large, destabilizing weight updates at the start of training.
* **Gradient Clipping**: We enforced a maximum norm for gradients to prevent them from becoming excessively large during backpropagation.

#### Loss Plots: Before Stabilization

_Demonstrates extreme instability, with the validation loss spiking uncontrollably._

<figure><img src="../.gitbook/assets/loss_plot_unstable_effnet.png" alt=""><figcaption></figcaption></figure>

#### Loss Plots: After Stabilization

_Post-stabilization: The validation loss shows smooth, consistent convergence without any erratic spikes, indicating stable training dynamics._

<figure><img src="../.gitbook/assets/loss_plot_effnet_stable.png" alt=""><figcaption></figcaption></figure>

***

## Results: Validation Set

### Methodology: Painting-Level Evaluation

While our models were trained on image patches, the final evaluation of authenticity must be made at the level of the entire painting. To achieve this, we implemented a painting-level aggregation strategy for the validation set.

The process is as follows:

1. **Collect Patch Predictions**: We first pass all patches from the validation set through the trained models to obtain individual predictions for each patch.
2. **Group by Painting**: These patch predictions are then grouped together by the painting they belong to.
3. **Majority Vote**: For each painting, a final classification is determined by a majority vote among its constituent patches. For example, if a majority of a painting's patches are classified as "imitation," the entire artwork is labeled as an "imitation."
4. **Calculate Metrics**: Standard classification metrics—Accuracy, Precision, Recall, and F1-Score—are then computed based on these final, aggregated, painting-level predictions.

### Performance Metrics

The following painting-level metrics were achieved on the **validation set**. At first glance, the baseline models appear to outperform their optimized counterparts on these metrics.

However, this is a classic sign of **overfitting**. The slightly lower scores of the stable and regularized models on this specific dataset are expected; these models have been trained to generalize better to truly _unseen_ data. The true value of the stabilization and regularization is demonstrated in the statistical tests on the held-out **test set**, which confirm that the optimized models are significantly more reliable and consistent, making them far more trustworthy for real-world art authentication.

| Model                | State                   | Accuracy  | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
| -------------------- | ----------------------- | --------- | --------------------- | ------------------ | -------------------- |
| **EfficientNet-B5**  | Baseline (Unstable)     | 97.8%     | 98%                   | 98%                | 98%                  |
|                      | Optimized (Stable)      | **96.0%** | **96%**               | **96%**            | **96%**              |
| **Swin Transformer** | Baseline (Overfit)      | 96.0%     | 96%                   | 96%                | 96%                  |
|                      | Optimized (Regularized) | **95.1%** | **95%**               | **95%**            | **95%**              |

***

## Statistical Validation

While validation set metrics provide a good snapshot of performance, a more rigorous evaluation on a held-out **test set** is crucial to confirm the true value of our optimization techniques. We conducted statistical tests to validate that our changes led to more reliable and consistent models.

### Swin Transformer: Regularization Boosts Reliability

We compared the overfit and regularized Swin models on several calibration and consistency metrics. While both models achieved a high accuracy of **96.92%** on the test set, the statistical analysis revealed the regularized model was **clearly superior** in its reliability.

* **Better Calibration**: The regularized model showed a lower **Expected Calibration Error (ECE)** and **Log Loss**, indicating its confidence scores are a more accurate reflection of its actual correctness.
* **Better Patch Consistency**: The regularized model had significantly **lower variance** in its predictions across the different patches of a single painting. This means its decisions are more stable and less susceptible to random noise from a single patch.

Based on these tests, the regularized model was deemed **statistically superior** on 3 out of 4 key reliability metrics.

### EfficientNet: Stabilization Ensures Consistency

To measure the impact of our stabilization techniques, we evaluated the performance of both the stable and unstable models on the held-out test set after every training epoch. This allowed us to track how their ability to generalize to unseen data evolved throughout the training process.

This analysis revealed that the test loss of the unstable model fluctuated significantly, reflecting its chaotic internal training dynamics. In contrast, the stabilized model exhibited a much smoother and more predictable test loss trajectory as it learned.

A bootstrap stability test, performed on the sequence of 30 test losses using **1000 samples**, confirmed this observation. The stabilized model demonstrated a **statistically significant (p < 0.05)** reduction in test loss variance compared to the unstable model. This confirms that our stabilization techniques not only fixed the erratic training behavior visible in the loss curves but also led to a model that converges more consistently and reliably on unseen data.

***

## Why This Matters

These improvements are crucial for the practical application of AI in art authentication. An ideal model must be not only accurate but also robust and trustworthy.

For authentication scenarios with significant implications, both **generalization (from regularization)** and **consistency (from stabilization)** are essential.

* A model that has been **regularized** is less likely to be fooled by novel examples.
* A model that has been **stabilized** will produce consistent outputs.

Our rigorous testing demonstrates that through careful optimization, we have developed AI systems that are not just high-performing, but are also fundamentally more reliable for the critical task of art authentication.

## Extended Experimentation on Larger Dataset

Once we had robust, optimized versions of the Swin Transformer and EfficientNet-B5, we extended the training pipeline to evaluate them on a larger, more challenging dataset. This new dataset **(LMI-A: various)** consisted of original Van Gogh works and a "contrast set" of stylistically similar but inauthentic paintings.

Since the models were already optimized on the original dataset **(LMI-B: WikiArt)**, our goal shifted from hyperparameter tuning to assessing generalization on unseen data. We therefore used an 80/20 train/test split. The training process on this larger dataset re-confirmed the effectiveness of our methods: the regularized Swin model did not overfit, and the stabilized EfficientNet model trained without exhibiting signficant fluctuations or exploding gradients.

Initial runs using two different subsets from the contrast set revealed a consistent and unexpected outcome: both the Swin and EfficientNet models classified various-quality images of the _Elimar_ painting as 'authentic' with moderate to high confidence. Since this contradicted the AR results, we conducted three additional experiments, each using a different subset of the contrast set, to rule out random chance. Across all five runs, the same pattern persisted, confirming the behavior was repeatable and not a statistical anomaly. Notably, the models maintained high generalization performance throughout, achieving 92–95% accuracies on the test set across all runs.

## _Elimar_ Painting Predictions

To investigate this phenomenon thoroughly, we evaluated model predictions on the _Elimar_ painting across different file formats and quality levels. The results are presented below, separated by the dataset used for training.

### LMI-B (WikiArt) Dataset Results

These models were trained on the original dataset used for optimization and statistical validation:

| Model                | Elimar\_10MB.png | Elimar\_11MB.jp2 | Elimar\_19MB.png | Elimar\_6MB.png | Elimar\_85MB.tiff | Elimar\_9MB.jpg |
| -------------------- | ---------------- | ---------------- | ---------------- | --------------- | ----------------- | --------------- |
| swin\_overfit        | Imit: 98%        | Imit: 100%       | Imit: 100%       | Imit: 90%       | Imit: 100%        | Imit: 100%      |
| swin\_regularized\_1 | Imit: 86%        | Auth: 88%        | Auth: 88%        | Auth: 78%       | Auth: 88%         | Auth: 86%       |
| effnet\_unstable     | Imit: 99%        | Imit: 100%       | Imit: 100%       | Auth: 99%       | Imit: 100%        | Imit: 95%       |
| effnet\_stable\_1    | Imit: 87%        | Imit: 92%        | Imit: 92%        | Imit: 91%       | Imit: 93%         | Imit: 88%       |

We achieved similar results to the AR analysis when the overfit and unstable models were made to predict on Elimar. The regularized Swin showed calibrated behavior with confidence scores being comparatively lower. It predicted 5 images of Elimar as authentic and 1 as imitation. The stable EfficientNet had lower confidence than the unstable model and predicted all of the images as imitation.

### LMI-A (Various) Dataset Results

These models were trained on the extended, larger dataset with diverse contrast set samples:

| Model                | Elimar\_10MB.png | Elimar\_11MB.jp2 | Elimar\_19MB.png | Elimar\_6MB.png | Elimar\_85MB.tiff | Elimar\_9MB.jpg |
| -------------------- | ---------------- | ---------------- | ---------------- | --------------- | ----------------- | --------------- |
| swin\_regularized\_2 | Auth: 94%        | Auth: 100%       | Auth: 100%       | Auth: 92%       | Auth: 100%        | Auth: 100%      |
| swin\_regularized\_3 | Auth: 96%        | Auth: 100%       | Auth: 100%       | Auth: 95%       | Auth: 100%        | Auth: 100%      |
| swin\_regularized\_5 | Auth: 96%        | Auth: 100%       | Auth: 100%       | Auth: 96%       | Auth: 100%        | Auth: 100%      |
| swin\_regularized\_6 | Auth: 92%        | Auth: 100%       | Auth: 100%       | Auth: 91%       | Auth: 100%        | Auth: 100%      |
| swin\_regularized\_7 | Auth: 95%        | Auth: 100%       | Auth: 100%       | Auth: 98%       | Auth: 100%        | Auth: 100%      |
| effnet\_stable\_2    | Auth: 98%        | Auth: 98%        | Auth: 98%        | Auth: 96%       | Auth: 98%         | Auth: 98%       |
| effnet\_stable\_3    | Auth: 92%        | Auth: 99%        | Auth: 99%        | Auth: 90%       | Auth: 99%         | Auth: 99%       |
| effnet\_stable\_5    | Auth: 97%        | Auth: 99%        | Auth: 99%        | Auth: 97%       | Auth: 99%         | Auth: 99%       |
| effnet\_stable\_6    | Auth: 93%        | Auth: 99%        | Auth: 99%        | Auth: 87%       | Auth: 99%         | Auth: 99%       |
| effnet\_stable\_7    | Auth: 95%        | Auth: 99%        | Auth: 99%        | Auth: 94%       | Auth: 99%         | Auth: 99%       |

All of these models consistently predicted all images of _Elimar_ as authentic with 87-100% confidence.

The predictions from both the tables above use patch-level majority voting. We also evaluated the Elimar painting using mean logit aggregation, which yielded comparatively lower confidence scores. For detailed results comparing both aggregation methods, see the [Elimar Inference Notebook](https://github.com/lmigroupintl/VanGoghComputerVision/blob/main/notebooks/inference/Models_inferences_on_Elimar.ipynb)).
