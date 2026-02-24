# Data & Data Processing

Schaerf et al. curated an authentic dataset inconsistently reported in the article as comprising 654 and 671, each validated against the authoritative La Faille _Catalogue Raisonné_ to ensure widespread acceptance of their authenticity.

The authors were careful to distinguish their work from prior studies on the basis of excluding any works with disputed provenance, in an attempt to minimize label noise. They did not, however, specify which paintings were included in their initial dataset nor which they excluded on the basis of questionable authenticity.

Nor was AR precise in the numbers of artworks the firm selected. They state in the same paper that they used 654/671 confirmed works by van Gogh, and then reduced this number to 137 paintings to narrow in on close imitations of van Gogh works; the _Wired_ article differently notes that this later experiment used 834 van Gogh and 1,785 non-van Gogh paintings. A 2024 paper by the firm (Ostmeyer et al., "Synthetic images aid the recognition of human-made art forgeries") does provide a training dataset for download, but the number of images again differ: 126 authentic and 321 inauthentic (with 13, or 10%, of the authentic works being drawings rather than paintings). This makes it impossible to reconstruct precisely which paintings AR trained on, and for which of their models.

Nor did the authors of Schaerf et al. outline the sources of their image data and what from these sources was excluded. The authors state that other models “are usually trained on images downloaded from WikiArt … (a less reliable source as compared to the established _Catalogue Raisonné_).” This fails to distinguish between downloading images from WikiArt and trusting WikiArt’s attributions, and it leaves unresolved the question of where the study sourced its digital images. This makes it impossible to reconstruct precisely what image data AR used to train the discussed models.

We organized our training dataset using images primarily sourced from WikiArt solely for the educational and research purposes of this code repository. The image data to which the repository points should be used only for these same non-commercial purposes. We have included a [spreadsheet](https://github.com/lmigroupintl/VanGoghComputerVision/blob/main/data/images.csv) in the code repository to aid in the precise reconstruction of our study for research purposes. This lists each painting’s title, image dimensions, whether it was used for training, validation, or test, and its URL — or the page on which it was found — is included wherever possible.

## Comparison of Datasets

LMI Group produced two image datasets in order to test rigorously the sensitivity of these models to differences in inputs. The comparison tables shown below allow users to see how our training data lines up with that used by AR.

### Authentic (van Gogh) Datasets

LMI Group's first collection of authentic van Gogh paintings was sourced by hand in the course of its research of _Elimar._ For this experiment, we also included oil paintings tagged by WikiArt as authentic works by van Gogh. This follows AR's reporting of several differently-sized image datasets in its published work.

<table><thead><tr><th>WIRED (2025)</th><th>Ostmeyer et al. (2024)</th><th>Schaerf et al. (2024)</th><th width="172">LMI-A (various)</th><th>LMI-B (WikiArt)</th></tr></thead><tbody><tr><td>834</td><td>126 (14 being drawings)</td><td>654/671 ("Standard") or 137 ("Refined")</td><td>884</td><td>754</td></tr></tbody></table>

### **Non-Authentic/Contrast Datasets**

LMI Group, following Schaerf et al.'s paper where possible, used (for the present research purposes) images from WikiArt that conformed closely to the authors' described "standard" and "contrast" datasets.

<table><thead><tr><th>WIRED article (2025)</th><th>Ostmeyer et al. (2024)</th><th>Schaerf et al. (2024)</th><th width="172">LMI-A (various)</th><th>LMI-B (WikiArt)</th></tr></thead><tbody><tr><td>1,785</td><td>321</td><td>669 ("Standard") or 137 ("Refined")</td><td>Random selections of 882-885 from a total of 2130 non-VG paintings.</td><td>Random selections of 753 from a total of 988 non-VG paintings.</td></tr></tbody></table>

We selected our contrast set from the works of the same artists named in the study; but the sub-categories articulated by the authors ("imitations," "forgeries," "in the style of van Gogh") were too vague to allow for a meaningful reconstruction where the authors were not named.

***

## Data Preprocessing / Patch Extraction

Following the methodology established by Schaerf et al. (2024), we implemented a patch-based preprocessing approach for high-resolution image segmentation. Images with minimum dimensions exceeding 1024 pixels were subdivided into 4×4 non-overlapping patches, while images with minimum dimensions ranging from 512 to 1024 pixels were partitioned into 2×2 patches. Binary classification labels were assigned to authentic and contrast datasets for subsequent model training.

Our implementation of the patch extraction process incorporates several robustness measures:

* Error Handling: Graceful management of corrupted or unreadable image files
* Empty Patch Filtering: Automatic removal of patches with insufficient content
* Metadata Preservation: Maintenance of painting identifiers for subsequent aggregation

Sample Weighting: In the standard contrast set, imitations are weighted more heavily than proxies (weight=10), as this improves results in preliminary experiments according to Schaerf et al. (2024).

Our system manages class imbalance by undersampling the contrast set. We reduce its size to target the number of authentic van Gogh paintings, which was 754 for the initial run and 884 for all other runs.

Further, our implementation uses Group-Aware Data Splitting: Dataset partitioning employs GroupShuffleSplit to ensure painting-level separation between training, validation and test sets, preventing data leakage and enabling realistic performance evaluation. This approach maintains the integrity of the authentication task by ensuring no patches from the same painting appear in training, validation and test sets.

### A Note on Image Quality

The quality of images in the art market currently significantly varies due to legacy imaging practices and a lack of general adherence to rigorous imaging standards. Variability in image quality will impact a CV model to the extent that the quality of training data will affect any AI.

Variability in image quality exists for a variety of reasons including resource allocation across art collections. Artworks by one artist may make up a larger component of the collection of one museum compared to that of another museum. This and differences in financial resources can lead to different image collection practices among museums and within a museum across different parts of its own collection. The art industry will also need to reach universal standards in image capturing, digital preservation (addressing such risks as digital rot), and standardization in downstream image duplication standards and image accessibility. Each of these variances individually and combined will influence how a CV model learns what visual data makes a van Gogh a van Gogh.&#x20;

The _Wired_ article and what AR otherwise discloses in its public data about its CV modelling highlights the need for data-driven approaches to art authentication to adhere to international standards. As recently as 2023, with the publication of a study on Cézanne’s _Boy in a Red Vest_, the AR researchers trained their model with images which fell into three tiers:

* Fewer than 512x512 pixels overall
* From 512x512 to 1024x1024 pixels overall
* More than 1024x1024 pixels overall

The interrogated image that was used in the 2023 study had higher overall dimensions, of 2500 x 3097 pixels (width x height). However, after factoring the size of the test painting of 80 cm x 64.5 cm (or 31.5 in. x 25.4 in (height x width), the effective overall resolution was only approximately 98-100 pixels per inch (PPI).

Put in other terms, the test image for the 2023 study of pixel resolution of 2500 x 3097 equals only roughly 8 megapixels, which is the equivalent of what could be captured by a 2011 iPhone 4S camera. This study concluded that the tested painting was by the attributed author to a probability of 89.58%. The poor image resolution reflected above would likely not enable the level of physical detail that would be needed to support such a confident attribution.

Reliable CV attributions will require image data that adheres to rigorous scientific standards. The current international standard for PPI in images of fine art is more than six times what was used in the discussed 2023 AR study, at 595 PPI for interrogated data and training data referencing the FADGI 4-star (ISO 19264-1 Level A) standard for large format digital reproduction.

Since CV is based on recognizing patterns in the pixels that make up digital images, it is important to understand how much data these images contain; PPI provides a good proxy indicator of this data-richness. The 2023 Cézanne study described above does not identify the PPI of the images used to train its model, but the three tiers of image size outlined in the paper allow us to set a ceiling on their quality—or, stated differently, to show how small the paintings in question would have to be in order to reach even 300PPI.

<table data-header-hidden><thead><tr><th valign="top"></th><th valign="top"></th><th valign="top"></th></tr></thead><tbody><tr><td valign="top"><p>Training image dimensions as reported</p><p>(but without disclosure of physical size of training data paintings)</p></td><td valign="top">Effective pixels of training data artworks assuming comparable size to interrogated Cezanne painting (2500 x 3097 reported pixels)</td><td valign="top">Effective size of training data artworks needed to match interrogated Cezanne painting @ 300 PPI (~50% of the FADGI standard)</td></tr><tr><td valign="top">&#x3C; 512 x 512 </td><td valign="top">20 PPI</td><td valign="top">&#x3C; 1.7”  x  1.7”</td></tr><tr><td valign="top">512 x 512 to 1025 x 1024</td><td valign="top">20 to 40 PPI</td><td valign="top">3.4” x 3.4”</td></tr><tr><td valign="top">> 1024 x 1024</td><td valign="top">300 PPI</td><td valign="top">8.3” x 8.3”</td></tr></tbody></table>

In short, although AR did not provide the dimensions of the paintings whose image data its models were trained on, its disclosed information indicates that either these paintings were largely below 50% of the FADGI 4-star reproduction standard, or they were all less than nine inches across, and most less than four--the latter scenario does not conform with Cézanne's ouevre. Similarly, in the 2024 Ostmeyer et al. study, which did disclose the PPI of its training dataset images, we find that only 32 of the 126 van Gogh works (or 25%) exceeded 300PPI.

Standards-based practices around data will need to take sufficient hold in the art industry for AI and CV in particular to be able to intelligently weigh in on questions of authentication.
