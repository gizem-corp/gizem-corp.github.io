---
layout: single
title: "RETFound: Learning From the Retina Without Labels"
date: 2026-07-05
permalink: /retfound/
tags:
  - medical-ai
  - self-supervised-learning
  - foundation-models
  - retinal-imaging
excerpt: "How masked autoencoders turn large collections of unlabelled retinal images into reusable clinical representations."
toc: true
toc_sticky: true
author_profile: false
share: false
comments: false
---

*How masked autoencoders turn large collections of unlabelled retinal images into reusable clinical representations.*

Retinal images are routinely collected in clinical care, but high-quality disease labels are expensive to obtain. RETFound asks whether a model can first learn from large amounts of unlabelled retinal data and then adapt efficiently to many different clinical tasks.

This blog post explains the main idea behind RETFound, how masked autoencoders are used for self-supervised learning, what the experiments demonstrate, and why important challenges remain before clinical deployment.

## The retina as a window into health

The retina is more than the light-sensitive tissue at the back of the eye. It is one of the few places in the human body where clinicians can non-invasively observe both neural tissue and a dense vascular network in vivo.

This makes retinal imaging valuable for detecting ocular disease. A retinal image can reveal local pathological changes such as haemorrhages, exudates, changes around the optic nerve, or structural alterations in retinal layers. These signs can support the diagnosis and monitoring of conditions including diabetic retinopathy, glaucoma, and age-related macular degeneration.

However, the retina may also provide indirect information about health beyond the eye. Retinal blood vessels are part of the circulatory system, while the optic nerve and retinal layers are closely connected to the central nervous system. Because of this, changes visible in retinal images may be associated with cardiovascular or neurodegenerative processes elsewhere in the body.

This emerging research area is often called *oculomics*: the use of retinal data to study systemic health. The idea is promising, but it should be interpreted carefully. A retinal image does not directly diagnose heart disease, stroke, or Parkinson's disease. Instead, it may contain visual patterns that help estimate future risk when combined with appropriate clinical data and validation.

RETFound is interesting because it attempts to learn these potentially useful retinal patterns without requiring disease labels during the first training stage. Rather than starting with a single clinical prediction task, it first learns a broad representation of retinal anatomy and image context.

## CFP and OCT: two complementary views of the retina

RETFound is trained separately on two common retinal imaging modalities: colour fundus photography, or CFP, and optical coherence tomography, or OCT.

### Colour fundus photography

CFP is a two-dimensional colour photograph of the retinal surface. It provides a wide view of visible structures such as the optic disc, the macula, and the retinal blood vessels.

Because CFP captures colour and vascular appearance, it can reveal clinically important signs such as haemorrhages, lipid deposits, vessel abnormalities, or changes around the optic nerve. These features are particularly relevant for diseases such as diabetic retinopathy and glaucoma.

A useful intuition is that CFP answers the question: *What does the retina look like from above?*

### Optical coherence tomography

OCT provides a different type of information. Instead of showing the retinal surface, it creates cross-sectional scans through the retina. This makes the layered internal structure of the tissue visible.

With OCT, clinicians can examine changes in retinal thickness, disruptions of individual layers, fluid accumulation, and structural abnormalities that may not be obvious in a surface-level photograph. OCT is therefore especially useful when disease affects the internal anatomy of the retina.

A useful intuition is that OCT answers the question: *What does the retina look like from the side?*

### Why use separate models?

CFP and OCT contain complementary but very different visual information. CFP images emphasize colour, vessels, and surface-level lesions, whereas OCT images emphasize depth and retinal microstructure.

For this reason, the RETFound paper trains separate foundation models for CFP and OCT rather than combining both modalities in one network. This allows each model to specialize in the visual patterns of its own modality.

At the same time, this is also a limitation of the current approach. A future multimodal model could potentially combine the broad vascular information from CFP with the detailed structural information from OCT.

## Why labels are the bottleneck in medical imaging

Modern hospitals generate large amounts of imaging data as part of routine care. Retinal scans are collected during screening, follow-up appointments, and disease monitoring. In principle, this should make ophthalmology an ideal domain for deep learning.

In practice, the limiting resource is not usually the image itself. It is the label.

A high-quality clinical label may require an ophthalmologist to inspect an image, identify pathological findings, compare the image with patient records, and assign a diagnosis or disease severity grade. For prognosis tasks, labels can be even more difficult to obtain because they require longitudinal follow-up: researchers need to know what happened to the patient months or years after the image was acquired.

This creates several challenges.

First, expert labels are expensive. Specialist time is limited, and large-scale image annotation can be difficult to justify in a busy clinical environment.

Second, labels are often task-specific. A dataset labelled for diabetic retinopathy may not include reliable labels for glaucoma, age-related macular degeneration, or future cardiovascular events. As a result, a model trained for one task cannot automatically be reused for another.

Third, clinical labels are not always perfectly clean or consistent. Different graders may interpret borderline cases differently, and datasets may use different diagnostic definitions, devices, or grading protocols.

A purely supervised model learns only from the labelled examples available for one specific task. If the dataset is small, biased, or narrowly defined, the model may learn representations that work well only in that setting.

> **The central opportunity is therefore to learn from the images before asking the model to solve a specific clinical task.**

This is where self-supervised learning becomes useful. Instead of requiring a clinician to tell the model what every image means, self-supervised learning creates a training signal directly from the image itself. The model can learn useful visual structure from large unlabelled datasets and later use a much smaller labelled dataset for task-specific adaptation.

For medical imaging, this is especially attractive. The raw data are often abundant, while expert annotations are scarce.

## From task-specific classifiers to retinal foundation models

Traditional medical image classifiers are usually trained for one narrow question.

For example, a model may be trained to answer:

- Does this image show diabetic retinopathy?
- Is this patient likely to have glaucoma?
- Does this OCT scan contain a sign of wet age-related macular degeneration?

These models can be useful, but each new task may require collecting a new labelled dataset and training a new model. This process does not scale well when the number of possible diseases, imaging devices, hospitals, and patient groups increases.

A retinal foundation model takes a different approach.

Instead of learning one task directly, the model is first pretrained on a very large collection of retinal images. During this stage, it is not optimized for a specific diagnosis. Its goal is to learn a broad representation of retinal anatomy, image structure, and visual context.

After this pretraining stage, the same encoder can be adapted to several downstream tasks with comparatively little labelled data.

In the RETFound paper, the downstream tasks cover three very different clinical settings:

1. **Diagnosis** — identifying diseases that are present in the image now, such as diabetic retinopathy or glaucoma.

2. **Prognosis** — predicting whether a future event may occur, such as conversion of the fellow eye to wet age-related macular degeneration within one year.

3. **Oculomics** — estimating the future risk of systemic diseases, including myocardial infarction, heart failure, ischaemic stroke, and Parkinson's disease.

This is the key promise of the foundation-model approach: one pretrained representation can support many specialized applications.

However, calling a model a foundation model does not mean that it solves every task equally well. A pretrained representation can improve transfer learning, but performance still depends on the downstream data, the patient population, the imaging device, the clinical outcome definition, and the degree of domain shift between training and deployment.

RETFound should therefore be understood as a reusable starting point. It is designed to reduce the amount of labelled data and training needed for new retinal tasks, rather than replacing clinical validation or eliminating the need for task-specific fine-tuning.

## Masked autoencoders: learning by reconstructing hidden content

The core method behind RETFound is a **masked autoencoder**, or MAE.

A masked autoencoder learns from an image by hiding a large part of it and asking the model to reconstruct what is missing. Unlike supervised learning, it does not need a disease label. The original image provides its own training signal.

![A masked autoencoder pipeline: a retinal image is split into patches, most patches are hidden, a Vision Transformer encoder processes visible patches, and a decoder reconstructs the missing content.](/images/retfound/mae-pipeline.png)
{: .align-center}

*Figure 1. The masked-autoencoder learning process used by RETFound. Illustration created for this post.*

### Step 1: Split the image into patches

Instead of processing the image as one large grid of pixels, the model divides it into small image patches. RETFound uses patches of size 16 × 16 pixels.

Each patch is treated similarly to a token in a language model. In natural language processing, a Transformer receives words or subwords as tokens. In a Vision Transformer, it receives image patches as visual tokens.

### Step 2: Hide most of the image

A large fraction of patches is removed before the image is given to the encoder.

For RETFound, the masking ratio is intentionally high:

- **75% of CFP patches are masked**
- **85% of OCT patches are masked**

This makes the reconstruction task difficult enough that the model cannot solve it only by copying nearby pixel patterns. To infer the missing content, it must learn broader image context, such as the continuity of blood vessels, the typical location of the optic disc, or the layered structure visible in OCT.

### Step 3: Encode the visible context

Only the visible patches are passed to the encoder. RETFound uses a large Vision Transformer encoder, specifically a ViT-Large model with 24 Transformer blocks and an embedding size of 1,024.

The encoder transforms the visible retinal patches into high-level feature representations. At this point, it does not predict a disease label. Its task is to summarize enough anatomical and visual context to make reconstruction possible.

### Step 4: Reconstruct the missing patches

A smaller Transformer decoder receives the encoded visible patches together with placeholder tokens representing the masked locations.

The decoder attempts to reconstruct the original image content. During pretraining, the model is optimized according to reconstruction quality on the hidden patches.

The important idea is not that a reconstructed retinal image must look visually perfect. The important idea is that reconstruction encourages the encoder to learn reusable information about retinal structure.

### Why is this useful for medical images?

Medical images contain recurring anatomical patterns. A healthy retina has characteristic vessel geometry, layer organization, and spatial relationships between structures. Disease may appear as a deviation from these normal patterns.

By repeatedly reconstructing heavily masked retinal images, the encoder is encouraged to learn this structure before it ever sees a downstream clinical label.

This creates a useful separation:

| Training stage | Input | Target | Main learning signal |
|---|---|---|---|
| Self-supervised pretraining | Partially masked retinal image | Missing image patches | Reconstruction quality |
| Downstream fine-tuning | Complete retinal image | Disease label or future outcome | Supervised classification objective |

The MAE objective therefore teaches the model **how retinal images are organized**, while downstream fine-tuning teaches it **which visual patterns matter for a specific clinical task**.

## RETFound's sequential pretraining strategy

RETFound does not learn retinal representations from scratch. Instead, it uses two consecutive self-supervised learning stages.

![Sequential pretraining diagram: self-supervised learning on ImageNet-1k, continued self-supervised learning on retinal images, then supervised fine-tuning for diagnosis, prognosis, and systemic risk prediction.](/images/retfound/sequential-pretraining.png)
{: .align-center}

*Figure 2. RETFound first learns generic visual structure from natural images and then specializes on retinal images before task-specific fine-tuning. Illustration created for this post.*

### Stage 1: self-supervised learning on natural images

The first stage starts with ImageNet-1k, a large dataset of natural images. The model learns generic visual concepts such as edges, texture, shape, object boundaries, and spatial structure.

Although natural images are very different from retinal scans, these general visual features can still provide a useful initialization for medical imaging.

### Stage 2: continued self-supervised learning on retinal images

The second stage continues MAE pretraining on retinal images.

The retinal dataset contains approximately 1.6 million unlabelled images:

- 904,170 colour fundus photographs
- 736,442 OCT scans

At this stage, the model specializes its generic visual representations to retinal anatomy, retinal image quality, vessel patterns, optic-disc appearance, and OCT layer structure.

### Why not train only on retinal images?

The paper compares RETFound with a retinal SSL model trained from scratch. This comparison is important because it tests whether natural-image pretraining contributes useful information beyond the retinal data itself.

The reported results suggest that it does.

RETFound generally performs better than both:

- a model pretrained only on natural images, and
- a model pretrained only on retinal images from scratch.

This supports the idea that generic visual knowledge and retina-specific knowledge are complementary.

However, this should not be interpreted as a universal rule for every medical-imaging problem. The benefit of sequential pretraining depends on the amount of domain data, the similarity between source and target images, the architecture, and the downstream task.

## From pretrained features to clinical predictions

After pretraining, RETFound is adapted to a particular clinical task using labelled data.

The reconstruction decoder is discarded. Only the pretrained ViT-Large encoder is retained, together with a multilayer perceptron classification head.

![Fine-tuning diagram: a complete CFP or OCT image enters the pretrained encoder, the decoder is discarded, and a classification head outputs disease probabilities.](/images/retfound/fine-tuning.png)
{: .align-center}

*Figure 3. During downstream adaptation, RETFound keeps the pretrained encoder and replaces the reconstruction objective with a task-specific classification objective. Illustration created for this post.*

The input is now a complete CFP or OCT image. The encoder converts it into a high-level representation, and the classification head predicts an outcome.

Depending on the task, the output can be:

- a current disease category, such as diabetic retinopathy or glaucoma;
- the probability of future conversion to wet age-related macular degeneration;
- the predicted risk of a systemic event within the next three years.

For multiclass tasks, the model predicts probabilities across several disease categories. For binary tasks, it predicts the probability of one clinical outcome occurring.

The fine-tuning stage uses labelled data and includes label smoothing, which softens the training targets slightly to reduce overconfidence and overfitting.

This setup is computationally practical because fine-tuning is much cheaper than pretraining a foundation model from scratch. According to the paper, pretraining required eight NVIDIA A100 GPUs for around two weeks, whereas a downstream fine-tuning experiment with 1,000 images could be run on a single NVIDIA T4 GPU in approximately 70 minutes.

## How was RETFound evaluated?

The authors evaluate RETFound across several tasks rather than relying on a single benchmark.

Their central comparison asks:

> Does sequential self-supervised pretraining on natural and retinal images produce more transferable representations than other common pretraining strategies?

To answer this, they compare four main models.

| Model | Pretraining strategy | What it tests |
|---|---|---|
| **SL-ImageNet** | Supervised learning on ImageNet-21k | Classical transfer learning from labelled natural images |
| **SSL-ImageNet** | Self-supervised learning on ImageNet-1k | Generic self-supervised visual features |
| **SSL-Retinal** | Self-supervised learning on retinal images from scratch | Retina-specific features without natural-image initialization |
| **RETFound** | SSL on ImageNet-1k, then SSL on retinal images | The combination of generic and retina-specific representation learning |

For the main experiments, the downstream fine-tuning procedure is kept comparable across models. This makes the pretraining strategy the main factor under investigation.

### Internal and external evaluation

The paper distinguishes between internal and external evaluation.

**Internal evaluation** means that the model is evaluated on a held-out test split from the same dataset used for fine-tuning.

**External evaluation** is more demanding. The model is fine-tuned on one dataset or cohort and tested on another dataset with different patients, clinical settings, cameras, or data distributions.

External evaluation matters because medical models often perform well in the environment where they were developed but degrade after being moved to a new hospital or population.

### Metrics

The main metrics are AUROC and AUPR.

- **AUROC** measures how well a model separates positive and negative cases across all possible decision thresholds.
- **AUPR** measures the trade-off between precision and recall and is particularly informative when positive cases are rare.

For multiclass tasks, the paper computes results for each class and then averages them.

Each experiment is repeated using five random seeds. The reported figures show mean performance and 95% confidence intervals across these runs. The authors also use two-sided t-tests to compare RETFound with the strongest competing baseline.

This statistical design is useful, but it is worth remembering what it does and does not measure. Repeating training with different seeds measures optimization and sampling variation within the experiment. It is not equivalent to validating a model in five independent real-world clinical trials.

## What did the experiments show?

The results can be summarized in three main findings.

![Results summary diagram: RETFound improves ocular transfer, shows stronger label efficiency, and often outperforms baselines on systemic-risk prediction while still suffering from external performance drops.](/images/retfound/results-summary.png)
{: .align-center}

*Figure 4. High-level summary of the main experimental findings and the remaining gap between research performance and clinical deployment. Illustration created for this post.*

### 1. Strong performance for ocular diagnosis and prognosis

RETFound performs best in most ocular-disease diagnosis datasets and in all reported cross-dataset diabetic-retinopathy evaluations.

The cross-dataset experiments are especially informative. Instead of training and testing within one dataset, the model is fine-tuned on one diabetic-retinopathy dataset and evaluated on another. This tests whether learned representations survive differences in acquisition conditions, populations, and data quality.

RETFound also achieves the strongest performance for one-year prediction of conversion to wet age-related macular degeneration in the fellow eye:

| Prognosis task | Input modality | RETFound AUROC |
|---|---:|---:|
| One-year wet-AMD conversion | CFP | 0.862 |
| One-year wet-AMD conversion | OCT | 0.799 |

These results suggest that the learned representation is useful not only for recognizing visible disease signs but also for a clinically meaningful prediction task.

### 2. Better performance on difficult systemic prediction tasks

The systemic-disease tasks are much more challenging than ocular diagnosis.

RETFound is evaluated for three-year incidence prediction of myocardial infarction, heart failure, ischaemic stroke, and Parkinson's disease. It generally improves on the compared baselines in internal evaluation and in most external evaluations.

For example, in the internal CFP setting, RETFound achieves AUROC values of:

| Systemic prediction task | RETFound AUROC with CFP |
|---|---:|
| Myocardial infarction | 0.737 |
| Heart failure | 0.794 |
| Ischaemic stroke | 0.754 |
| Parkinson's disease | 0.669 |

These numbers should be interpreted carefully.

The fact that RETFound ranks above the baselines is meaningful evidence that retina-specific pretraining helps. But relative improvement alone is not enough to establish clinical usefulness. The absolute performance is moderate for several tasks, and performance decreases when the model is evaluated on the external UK Biobank cohort.

In other words, retinal images may contain signals related to systemic risk, but this does not mean that the model is ready to make high-stakes medical decisions independently.

### 3. Better label efficiency

The label-efficiency experiments are one of the paper's most practical contributions.

Instead of using all labelled training examples, the authors fine-tune models with different fractions of the available data. RETFound performs particularly well when labels are scarce.

For heart-failure and myocardial-infarction prediction, RETFound outperforms the comparison models even when trained with only 10% of the labelled training data.

This does not mean that the remaining 90% of labels are unnecessary in every setting. Rather, it suggests that the pretrained encoder already captures useful retinal features, so fewer task-specific labels are needed to reach a given performance level.

The paper also reports faster adaptation. For the myocardial-infarction task, RETFound reaches convergence much earlier than the strongest baseline, corresponding to a potential training-time saving of around 80%.

Together, the label-efficiency and adaptation-efficiency results support the central value proposition of foundation models in medicine: expensive labels can be reserved for the final task-specific adaptation step instead of being required from the beginning.

### A note on the MAE-versus-contrastive comparison

The paper also compares MAE with contrastive self-supervised approaches including SimCLR, SwAV, MoCo-v3, and DINO.

MAE performs best in most reported downstream tasks. This is useful evidence that reconstruction-based learning works well in this setting.

However, the authors appropriately caution against interpreting the comparison as absolute proof that MAE is always superior. The approaches differ not only in their learning objective but also in architecture and hyperparameter choices. For example, some contrastive baselines use ResNet-based encoders while others use Transformers.

The safest conclusion is therefore:

> In this experimental setup, MAE was the strongest tested self-supervised strategy for RETFound, but perfectly controlled comparisons would be needed to isolate the effect of the objective itself.

## What did RETFound actually learn?

A central challenge in medical AI is understanding whether a model uses clinically meaningful information or exploits shortcuts that happen to correlate with the label.

The paper provides two forms of qualitative evidence.

### Reconstruction of retinal structure

First, the MAE reconstruction examples show that RETFound can recover major retinal anatomy even when most image patches are hidden.

For CFP images, the model reconstructs structures such as large vessels and the optic nerve. For OCT images, it reconstructs retinal layers including the retinal nerve fibre layer and retinal pigment epithelium.

This supports the claim that pretraining captures retina-specific visual context.

At the same time, successful reconstruction is not direct proof that the representation is clinically optimal. A model may reconstruct visually plausible structure without necessarily learning the causal features needed for every downstream disease task.

### Relevance maps during prediction

Second, the authors use RELPROP, a relevance-propagation method for Transformer models.

For ocular disease prediction, relevance maps highlight clinically plausible regions:

- haemorrhages and hard exudates for diabetic retinopathy;
- regions around the optic nerve for glaucoma;
- retinal structures and layers for several OCT-based tasks.

For systemic prediction, the highlighted regions include retinal vasculature, the optic nerve, and retinal layers.

These patterns are encouraging because they align with known clinical hypotheses. However, relevance maps should not be treated as complete explanations.

A heatmap can show which regions contributed to a prediction, but it cannot prove that the model reasons like a clinician, that the highlighted region caused the prediction, or that the model would behave safely under all clinical conditions.

### Age as a potential confounder

The paper also investigates age-related confounding for myocardial-infarction prediction.

Age is strongly related to many systemic diseases and may also influence retinal appearance. A model could therefore achieve apparently good performance simply by recognizing age-related visual changes rather than disease-specific signs.

The authors compare disease cases with control groups of different average ages. They show that a simple logistic-regression model using age alone performs well when the age gap between cases and controls is large.

As the age distributions become more similar, RETFound remains more stable than the other models. This is a useful analysis because it shows that the authors considered an important confounder.

Still, it does not eliminate all possible confounding. Retinal images can encode many correlated signals, including age, ethnicity, imaging device, coexisting diseases, and healthcare-access patterns.

## Limitations and future directions

RETFound is a strong research foundation model, but several limitations should be kept in mind.

### Development data are geographically concentrated

Most development images come from UK cohorts. Although the paper includes public datasets from several countries for ocular-disease evaluation, the large-scale pretraining data are not globally representative.

A broader and more balanced pretraining dataset would be needed to understand whether performance transfers equally well across geographic regions, ethnic groups, camera systems, and clinical workflows.

### External performance drops remain substantial

RETFound usually performs better than the comparison models on external evaluation, but the paper still reports meaningful performance drops when moving from the MEH-AlzEye cohort to UK Biobank.

This is not a minor technical detail. It is one of the most important findings for clinical translation.

A model can be relatively more robust than a baseline and still be insufficiently robust for deployment. Future work should test RETFound prospectively across independent hospitals, countries, cameras, and patient populations.

### CFP and OCT are not combined

CFP and OCT provide complementary information, but RETFound trains separate models for each modality.

A future multimodal model could combine:

- vascular and colour information from CFP;
- detailed retinal-layer structure from OCT;
- non-image information such as age, sex, visual acuity, medical history, and laboratory measurements.

This may improve performance, but it would also introduce new challenges related to missing modalities, data harmonization, privacy, and fairness.

### Systemic prediction remains an ambitious use case

The systemic-disease tasks are scientifically interesting, but they are also the easiest to overinterpret.

The model predicts future incidence or risk from retinal images. It does not diagnose systemic disease directly, and it does not demonstrate a causal biological mechanism.

For high-stakes outcomes such as myocardial infarction or stroke, retinal AI should be evaluated as one possible component of a broader clinical risk-assessment workflow, not as a standalone decision-maker.

### Pretraining is expensive

RETFound makes downstream fine-tuning cheaper, but the original pretraining stage is still resource-intensive.

The authors report using eight A100 GPUs for about two weeks. This creates a practical tension: foundation models may democratize downstream experimentation, but building the original foundation model remains accessible mainly to institutions with substantial data and compute resources.

### From retrospective evaluation to clinical impact

Finally, the paper is based on retrospective datasets. Before clinical deployment, a model would need prospective validation in realistic workflows.

Important questions include:

- Does the model improve clinician decisions?
- Does it improve patient outcomes?
- Does it remain calibrated after deployment?
- How should uncertainty be communicated?
- What happens when images are low quality or incomplete?
- Which patients benefit, and which groups may be disadvantaged?

These questions go beyond AUROC and are essential for responsible clinical AI.

## Takeaways

RETFound demonstrates a compelling use of self-supervised learning for medical imaging.

Its central contribution is not simply a new classifier for one disease. Instead, it shows how large collections of unlabelled retinal images can be converted into reusable representations that support many downstream tasks.

The paper provides evidence for three main claims:

1. **Sequential self-supervised pretraining on natural images and retinal images improves transfer performance.**

2. **The learned representations can reduce the amount of labelled data and fine-tuning time required for several clinical tasks.**

3. **Retinal foundation models are promising for ocular diagnosis, prognosis, and systemic-risk prediction, but external robustness and clinical validation remain major open challenges.**

The most balanced interpretation is that RETFound is an important step toward general-purpose retinal AI. It is a strong research foundation, not yet a finished clinical decision system.

## References

1. Zhou, Y. et al. (2023). *A foundation model for generalizable disease detection from retinal images*. Nature, 622, 156–163. https://doi.org/10.1038/s41586-023-06555-x

2. He, K. et al. (2022). *Masked autoencoders are scalable vision learners*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16000–16009. https://doi.org/10.1109/CVPR52688.2022.01553

3. Dosovitskiy, A. et al. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale*. International Conference on Learning Representations.

4. Bommasani, R. et al. (2021). *On the opportunities and risks of foundation models*. arXiv:2108.07258. https://doi.org/10.48550/arXiv.2108.07258

5. Chefer, H., Gur, S., & Wolf, L. (2021). *Transformer interpretability beyond attention visualization*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 782–791. https://doi.org/10.1109/CVPR46437.2021.00084
