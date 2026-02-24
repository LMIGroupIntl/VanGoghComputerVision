# Project background

This document outlines a 2025 reimplementation of the methodology described in "Art authentication with vision transformers" by Schaerf et al. (2024). The original study investigated the efficacy of vision transformers (ViTs), with particular emphasis on the Swin-Tiny transformer architecture, for enhancing the reliability of computer-based artwork authentication systems, specifically in reference to the problem of determining whether a painting was painted by Vincent van Gogh.

LMI Group set out to replicate the model built by AR by closely following the methods and dataset described in their 2024 paper. The goal was to understand how AR’s system reached the conclusion, announced in _Wired_, that there was a 97% probability that LMI Group was incorrect in its attribution of the van Gogh painting _Elimar_.

LMI Group followed AR’s methodology precisely as AR described it in the 2024 paper. At several junctures, we had to make reasonable assumptions due to unclear details in the original study. Below, we explain how we recreated the process described in that paper, and where we were required to make judgement calls, filling in gaps in keeping with best practices.

* **Data acquisition**: We compare AR’s training dataset, as described, to ours. AR did not disclose the source of its image data; for these educational and research purposes, we sourced our training images from multiple public repositories, including WikiArt (which aligns closely with images available from Wikimedia Commons). Our selection process and documentation are detailed below.
* **Data preprocessing**: We describe in detail how we reconstructed AR's patch analysis method. This meant dividing images into uniform squares for closer examination, and explaining how we handled any gaps or overlaps. Our approach in this area represents the scientifically best approach to replicating the methods described in the AR 2024 paper.
* **Training**: We describe our split between training, validation, and test sets, and our means of scoring paintings that have been sliced up into patches in the manner described above. There is likely little to no difference between AR’s pipeline and our own in this area. We set the number of epochs to 30, as the models converges by this point, and extending training further may lead to overfitting. AR did not specify this. As a result, our models may differ from its models in this aspect.
* **Evaluation**: We describe the process by which we evaluated the model, and the results we obtained.

These differences, however small, are important to highlight because LMI Group’s model determined, with a probability of 96%, that _Elimar_ is in fact an autograph painting by van Gogh. AI models are very sensitive to changes in training data, especially for smaller datasets. However, this degree of sensitivity is hard to explain and is a major source of uncertainty in the field.

To support transparency and reproducibility, we have published the source code and documentation for this experiment, so that others can review our methods and their relevance to fine art authentication.
