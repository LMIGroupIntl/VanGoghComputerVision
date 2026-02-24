# Discussion and Conclusion

Our reimplementation achieves similar accuracy and ROC-AUCs as Schaerf et al.

Our dataset collection methodology is the most difficult to validate against the comparison study, due to the insufficient documentation by the study in this area. However, the similarity in the number and categories of paintings, and our documentation of these and the image dimensions of each, should promote reproducibility and critical analysis in the field.

Our data preprocessing almost certainly reproduces the comparison experiment methodology, as this patch-based analysis is standard in the discipline and the study’s description of the method leaves little room for invention or guesswork.

Finally, training data split and epoch calibration we employed to avoid fine-tuning follow best practices and so should not be a source of significant disagreement in outputs.

What is particularly striking, then, is that our model, which followed the prior study’s design as closely as their published work would allow, came to precisely the opposite conclusion about _Elimar_. Using the same publicly-available digital image of the painting that was made available in early 2025, we found that the model output a strong 96% probability score in favor of van Gogh authorship – a mirror image of AR’s 97% score against, as reported in _Wired_.

This raises several questions about the design of the prior study:

* How sensitive is this method to minor variations in training data, for instance,
  * to the differences between the sets of 654 vs. 671 van Gogh images enumerated at different points in the 2024 paper, or
  * to the incredibly low number of images (137) used in their "refined" training dataset, or
  * to the massive changes in the training dataset (834 van Goghs) which _Wired_ reported was used in evaluating _Elimar_?
* How many epochs were used in the prior experimental study, and how did the training behavior of the two models compare: was EfficientNet unstable and did the SWIN model show signs of overfitting?
* Were there any unexpected deviations in the data processing, training, or evaluation pipeline, that would lead to such a striking difference in outputs?

Without publication of the full training datasets and codebases for the peer-reviewd 2024 study and the experiment run for the _Wired_ article, these questions are ultimately insoluble. LMI Group has sought to take a step towards transparency and reproducibility in this area by publishing, in this repo, in tandem with [its whitepaper on the subject](https://www.lmigroupintl.com/pdf/potential-role-ai-art-authentication):

* The model described above which was used to validate _Elimar_
* The codebase used to generate the model
* Documentation for the use of this code
* A registered list of the training data and its provenance

### Conclusion

LMI Group’s conclusion from these experiments is that while AI remains a promising technology for the scientific analysis of artworks in and beyond the domain of image data, at least in the context of authenticating artworks, the technology is currently too sensitive to overfitting, to the selection of training data, and to training data that is for various reasons too variable in quality.

The publication of the repeatable steps by which LMI Group came to this conclusion aims to open up this discussion around the use of AI in art authentication to broader and more rigorous inquiry. If AI is to be used in the high-stakes pursuit of fine art authentication, then such rigor and transparency will be key to validating its accuracy and advancing its development.
