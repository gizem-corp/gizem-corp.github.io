---
layout: single
title: "RETFound: Learning Retinal Representations From What Is Missing"
date: 2026-07-05
permalink: /retfound/
tags:
  - medical-ai
  - self-supervised-learning
  - masked-autoencoders
  - foundation-models
  - retinal-imaging
excerpt: "A technical reading of RETFound: how sequential masked-autoencoder pretraining turns unlabelled retinal images into transferable representations."
toc: true
toc_sticky: true
author_profile: false
share: false
comments: false
mathjax: true
classes: wide
---

*Can a model learn clinically useful retinal representations before it is ever told what a disease label means? RETFound addresses this question with sequential masked-autoencoder pretraining on natural and retinal images.*

> **Scope of this post.** This is not a section-by-section summary of Zhou *et al.* Instead, it reads RETFound as a scientific experiment about representation learning: what is the learning problem, what does each comparison isolate, and how far do the reported results justify the clinical claims?

## The question behind RETFound

Retinal imaging is an unusually attractive setting for medical machine learning. Colour fundus photographs and optical coherence tomography scans are routinely acquired in eye care, creating large image repositories long before anybody decides which downstream prediction task they may support. Yet the labels needed for conventional supervised learning are much scarcer than the images themselves. A reliable diabetic-retinopathy grade, a glaucoma assessment, or a future cardiovascular outcome may require specialist review, longitudinal records, or both.

This imbalance creates a familiar problem. A task-specific classifier can be excellent when it is trained on a well-labelled dataset collected for exactly that purpose. It is much less obvious how to reuse the same model for another disease, another imaging device, or another hospital. RETFound asks whether representation learning can move the expensive clinical supervision to the *end* of the pipeline: first learn a broad visual model of the retina from unlabelled images, then adapt that model with comparatively few labels to several clinical tasks.[^retfound][^foundation]

The paper therefore tests a stronger hypothesis than “self-supervision helps”. Its central claim is that **sequential self-supervised pretraining—first on natural images, then on retinal images—produces representations that transfer better than either source of supervision alone**. The downstream evidence spans three increasingly difficult settings:

1. **Diagnosis:** detecting disease that is already visible in the image;
2. **Prognosis:** predicting whether a fellow eye will convert to wet age-related macular degeneration (AMD) within one year;
3. **Oculomics:** estimating the three-year incidence of systemic conditions from retinal images.

The third setting is especially important to phrase carefully. A retinal scan does not directly diagnose heart failure, myocardial infarction, stroke, or Parkinson’s disease. RETFound outputs a statistical risk estimate learned from associations in retrospective data; it does not establish a causal mechanism and it should not be interpreted as a stand-alone clinical decision-maker.

<!-- FIGURE PLACEHOLDER 1
Create an original overview figure here:
Retinal images -> self-supervised representation learning -> three downstream task families
Diagnosis / prognosis / systemic-risk prediction.
Caption: "Figure 1. RETFound turns routine retinal images into a shared starting point for several task-specific prediction models."
-->

## Retinal images are not one kind of data

RETFound trains separate models for two modalities with very different visual structure.

### Colour fundus photography: a surface view

A colour fundus photograph (CFP) is a two-dimensional image of the retinal surface. It shows large blood vessels, the macula, the optic disc, and visible lesions such as haemorrhages or lipid exudates. This makes CFP particularly useful for diseases in which colour, vascular geometry, or surface-level pathology are informative.

### Optical coherence tomography: a cross-sectional view

Optical coherence tomography (OCT) provides a cross-sectional image through the retina. It reveals the layered microstructure of the tissue and can expose changes in thickness, fluid, or layer integrity that a surface photograph may miss.

A simple mental model is useful: **CFP asks what the retina looks like from above; OCT asks what it looks like in cross-section.** Their information is complementary, but the paper does not fuse them. Instead, it trains one RETFound model for CFP and another for OCT.

That separation is methodologically sensible—the images have different appearance, dimensionality, and disease cues—but it is also a limitation. A multimodal model could potentially combine vascular information from CFP with layer-level structure from OCT. Whether this would improve the same clinical tasks remains untested here.

### What reaches the network?

The preprocessing choices matter because they define the problem that the model is actually allowed to solve.

For CFP, the authors use AutoMorph to remove the background and retain the retinal field. For OCT, they extract the middle slice from the scan rather than modelling the full volumetric acquisition. Both modalities are resized to 256 × 256 pixels, then augmented with random crops that are resized to 224 × 224, random horizontal flips, and normalization.[^retfound]

The OCT decision deserves attention. Taking a central B-scan makes a large-scale Transformer pipeline tractable, but it discards information away from that slice. The reported OCT results should therefore be read as results for **single-slice OCT representation learning**, not as evidence that the entire 3D scan has been exploited.

## Formulating RETFound as a machine-learning problem

A useful way to understand RETFound is to separate its two objectives. The model solves one self-supervised image-reconstruction problem during pretraining and one supervised prediction problem during adaptation.

| Stage | Input | Output / target | Learning objective |
|---|---|---|---|
| **Masked-autoencoder pretraining** | A retinal image with many patches hidden | Reconstruction of the missing patches | Minimize reconstruction error on masked content |
| **Task-specific fine-tuning** | A complete CFP or OCT image | Disease class or event probability | Minimize supervised classification loss with label smoothing |

### Stage 1: reconstructing masked retinal content

Let a preprocessed image be denoted by $x$. After the 224 × 224 crop, RETFound divides it into 16 × 16 patches, producing a sequence of $N = 14 \times 14 = 196$ visual tokens. Let $M$ be the set of masked patch indices and $V$ the visible indices.

The encoder receives only the visible patches:

$$
z = E_{\theta}(x_V)
$$

where $E_{\theta}$ is the Vision Transformer encoder. A decoder then receives the encoded visible representation together with mask tokens and estimates the missing content:

$$
\hat{x}_M = D_{\phi}(z, M)
$$

Conceptually, the pretraining loss can be written as

$$
\mathcal{L}_{\mathrm{MAE}}
=
\frac{1}{|M|}
\sum_{i \in M}
\ell(\hat{x}_i, x_i)
$$

where $\ell$ is a reconstruction-error term evaluated only on the hidden patches. The paper describes the objective as reconstruction error; the equation above is a compact way to state the masked-patch objective rather than an additional modelling assumption.[^retfound][^mae]

This is the first place where the “self” in self-supervised learning becomes concrete: the image supplies its own target. No clinician has to annotate the pretraining corpus.

### Stage 2: predicting a labelled clinical outcome

For a downstream task, the decoder is discarded. The retained encoder maps a full image $x$ to a representation $h$, and a multilayer perceptron head $g_{\psi}$ maps that representation to class probabilities:

$$
h = E_{\theta}(x), \qquad
p(y \mid x) = g_{\psi}(h)
$$

For a multiclass task, a label-smoothed version of the one-hot target $\tilde{y}$ is used in a cross-entropy-style objective:

$$
\mathcal{L}_{\mathrm{FT}}
=
-\sum_{c=1}^{C} \tilde{y}_{c} \log p(y=c \mid x)
$$

For binary prognosis and incidence tasks, the same logic reduces to estimating a probability for the event of interest. The authors train the downstream models for 50 epochs and select the checkpoint with the highest validation AUROC.[^retfound]

This two-stage formulation is the key conceptual distinction:

- Pretraining asks: **What visual structure is needed to complete a retinal image?**
- Fine-tuning asks: **Which parts of that learned representation help predict this particular clinical label?**

## Learning anatomy from missing patches

RETFound uses a masked autoencoder (MAE), a self-supervised architecture introduced for image representation learning by He *et al.*[^mae] The model deliberately hides most of its input, processes only the visible part with a large encoder, and reconstructs the image with a lighter decoder.

<!-- FIGURE PLACEHOLDER 2
Create an original MAE diagram here:
224x224 CFP/OCT -> 196 patches -> random masking -> visible patches to ViT-Large encoder -> latent tokens + mask tokens -> ViT-Small decoder -> reconstructed masked patches.
Indicate 75% masking for CFP and 85% for OCT.
Caption: "Figure 2. RETFound's masked-autoencoder objective. The encoder sees only the visible patches; reconstruction loss is computed on masked patches."
-->

### Why mask so aggressively?

RETFound masks 75% of CFP patches and 85% of OCT patches.[^retfound] At first this may sound counterintuitive: why make the input so incomplete?

The reason is to prevent easy local shortcuts. If only a tiny region were removed, a decoder might restore it by interpolating nearby texture. With most of the image hidden, the model has to use longer-range retinal regularities: vessel continuity, typical optic-disc position, the geometry of the fundus, or the layered organization of OCT.

This does **not** mean that the model learns medical reasoning in a human sense. It means that good reconstruction becomes difficult without representing recurring anatomical context. Whether that context is also useful for disease prediction is an empirical question—the remainder of the paper is designed to test it.

### Architecture and training scale

The encoder is the large Vision Transformer variant,[^vit] with 24 blocks and 1,024 features per token. Reconstruction is handled by a smaller eight-block decoder with 512-dimensional token features. The retinal stage uses 800 training epochs, a batch size of 1,792, and eight NVIDIA A100 GPUs; the authors report roughly fourteen days of pretraining time.[^retfound]

| Component | RETFound configuration |
|---|---|
| Encoder | ViT-Large; 24 blocks; 1,024 features per token |
| Decoder | Smaller ViT; 8 blocks; 512 features per token |
| Patch size | 16 × 16 pixels |
| CFP masking ratio | 75% |
| OCT masking ratio | 85% |
| Retinal SSL corpus | 904,170 CFP images + 736,442 OCT images |
| Retinal SSL compute | 8 × A100 GPUs, approximately 14 days |

The scale is both a strength and a caveat. It makes a rich retinal representation plausible, but it also means that training a comparable foundation model from scratch is not equally accessible to every clinical research group.

## Why the pretraining order is itself an experiment

The most interesting part of RETFound is not merely its architecture. It is the four-way comparison used to ask *where useful representations come from*.

| Model | Pretraining route | Main question addressed |
|---|---|---|
| **SL-ImageNet** | Supervised learning on ImageNet-21k | How strong is conventional transfer from labelled natural images? |
| **SSL-ImageNet** | Self-supervised learning on ImageNet-1k | How far do generic SSL image representations transfer without retinal specialization? |
| **SSL-Retinal** | SSL on retinal images from random initialization | Is retinal-domain SSL sufficient on its own? |
| **RETFound** | SSL on ImageNet-1k, then SSL on retinal images | Does generic SSL initialization add value before retinal specialization? |

The cleanest comparison is **SSL-Retinal versus RETFound**. Both learn from retinal images with SSL, but RETFound begins from an ImageNet-SSL representation. If RETFound is better, this supports the idea that generic visual structure helps the later retinal stage.

The other comparisons require more care.

- **SL-ImageNet versus SSL-ImageNet** changes both the supervision type *and* the source dataset: SL-ImageNet uses ImageNet-21k with roughly 14 million labelled images, whereas SSL-ImageNet uses ImageNet-1k with about 1.4 million images. It is therefore not a perfectly controlled “supervised versus self-supervised” ablation.
- **MAE versus contrastive SSL** is also not a pure objective-only comparison. The paper substitutes SimCLR, SwAV, MoCo-v3, and DINO into the framework, but their recommended architectures and hyperparameters differ. For example, SimCLR and SwAV use ResNet-based models while other variants use Transformers.[^retfound]

This does not invalidate the experiments. It simply changes the appropriate conclusion. The paper provides strong evidence that MAE is the best-performing *tested configuration* in this study. It does not isolate the MAE objective so completely that one can claim it is universally superior to contrastive learning.

<!-- FIGURE PLACEHOLDER 3
Create an original 2x2 ablation-logic diagram:
Rows: natural-image stage absent/present
Columns: retinal SSL absent/present
Place SSL-ImageNet, SSL-Retinal, RETFound; show SL-ImageNet separately as supervised reference.
Caption: "Figure 3. The four baselines are best read as an ablation of representation sources, not as a single binary comparison."
-->

## Evaluation: which claims are actually tested?

RETFound is evaluated on more than one kind of experiment, and each experiment supports a different claim.

| Claim | Experimental test | What the test can support | What it cannot establish |
|---|---|---|---|
| Better ocular transfer | Cross-dataset diabetic-retinopathy evaluation | Robustness across selected retinal datasets | Universal robustness to every clinic or country |
| Better future-risk prediction | One-year wet-AMD and three-year systemic-incidence tasks | Predictive association within the defined cohorts | Causality or clinical benefit |
| Better label efficiency | Fine-tuning with 10–100% of labels | Lower label demand for selected tasks | That labels are unnecessary |
| Better calibration | Reliability diagrams and expected calibration error | Better probability agreement in retrospective test data | Safe deployment without monitoring |
| Clinically meaningful representations | Reconstructions and RELPROP maps | Plausible anatomical or pathological focus | A complete explanation of the model's reasoning |

### Data sources and task families

The pretraining corpus contains 904,170 CFP images and 736,442 OCT images. The dominant source is MEH-MIDAS, a retrospective Moorfields collection of imaging records from people with diabetes; EyePACS contributes additional CFP data and a public OCT dataset contributes the remaining OCT data.[^retfound]

For ocular diagnosis, the authors use public datasets covering diabetic retinopathy, glaucoma, and multi-category retinal disease. For prognosis and systemic prediction, they use the MEH-AlzEye record-linkage cohort. The systemic external evaluation transfers models trained on MEH-AlzEye to UK Biobank.[^retfound]

This distinction matters. Some of the ocular tasks are external across countries and public datasets. The systemic “external” evaluation is a genuine cohort shift, but both source and target remain UK-based. It should not be equated with worldwide validation.

### Metrics and uncertainty

The paper reports AUROC and AUPR.

- **AUROC** summarizes ranking ability across all classification thresholds.
- **AUPR** is particularly informative when positive cases are uncommon, because it focuses on the precision–recall trade-off.

Each task is trained with five random seeds. The reported 95% confidence intervals are calculated from variation across these five training replicas, and two-sided t-tests compare RETFound with the strongest competing model.[^retfound]

This is useful for assessing sensitivity to optimization randomness. It is not the same as patient-level uncertainty or repeated validation across five independent hospitals. In particular, seed-based confidence intervals can look very narrow even when the more important uncertainty is the shift from one clinical environment to another.

## What the experiments support—and where the evidence stops

### Ocular diagnosis: the strongest transfer evidence

The cross-dataset diabetic-retinopathy experiments provide some of the clearest evidence that RETFound learns transferable features. A model fine-tuned on one dataset is evaluated on another rather than merely on a held-out split from the same source.

For example, when fine-tuned on APTOS-2019, RETFound achieved AUROC 0.822 on IDRiD and 0.738 on MESSIDOR-2. The paper reports that RETFound ranked first in all cross-evaluations among the compared models.[^retfound]

This is a stronger result than a conventional random train–test split because acquisition conditions, label protocols, and population characteristics differ between datasets. Still, it is transfer across a limited set of retinal benchmarks, not evidence that the model will work unchanged on arbitrary fundus cameras or care pathways.

### Prognosis: predicting wet-AMD conversion

For one-year prediction of wet-AMD conversion in the fellow eye, RETFound reaches an AUROC of 0.862 with CFP and 0.799 with OCT.[^retfound]

This task is clinically interesting because it asks the model to predict a future event rather than recognize an already visible diagnosis. It also illustrates why “fellow eye” matters: the input comes from the eye that has not yet converted, so the model is searching for risk-associated patterns before the target event occurs.

The result is encouraging, but it remains an internal evaluation on the AlzEye cohort. Prospective validation would be needed to determine whether the prediction changes surveillance, treatment timing, or patient outcomes.

### Oculomics: useful signal, difficult interpretation

The systemic tasks are the most ambitious and the easiest to overstate. In internal CFP evaluation, RETFound reports AUROCs of 0.737 for myocardial infarction, 0.794 for heart failure, 0.754 for ischaemic stroke, and 0.669 for Parkinson’s disease.[^retfound]

These outcomes show that the learned retinal representation improves relative ranking over the baselines in the paper. They do not show that a retinal image alone is sufficient for high-stakes systemic diagnosis.

The external results reinforce this caution. Performance decreases after transfer to UK Biobank, and for external ischaemic-stroke prediction RETFound and SSL-Retinal perform similarly rather than showing a clear advantage for sequential pretraining.[^retfound] The paper attributes part of this challenge to shifts in demographics, ethnicity, and imaging devices. It reports an AUROC drop for RETFound's stroke prediction of 0.16 with CFP and 0.19 with OCT between internal and external settings.[^retfound]

The scientifically balanced conclusion is therefore: retinal images appear to carry predictive information about systemic risk, and retinal SSL helps access that information. The paper does not demonstrate a causal pathway from retinal appearance to systemic disease, nor a ready-to-deploy screening tool.

### Label efficiency: a practically important result

RETFound’s most immediately useful contribution may be the label-efficiency analysis. The authors fine-tune models using different fractions of the available labels and find that RETFound outperforms the comparison models with only 10% of the labelled data for heart-failure and myocardial-infarction prediction.[^retfound]

This is not a statement that 90% of labels can always be discarded. The result is task-specific and depends on the target performance level. What it does show is that a pretrained retinal encoder can shift part of the learning burden away from downstream clinical annotation.

The adaptation-efficiency experiment points in the same direction. For myocardial-infarction prediction, RETFound reaches the selected validation checkpoint substantially earlier, corresponding to an estimated 80% reduction in training time relative to the strongest comparison model; for diabetic retinopathy on MESSIDOR-2, the corresponding estimate is 46%.[^retfound] These are convergence-time estimates under the reported training setup, not a universal promise of identical wall-clock savings.

### Does the model attend to medically meaningful structures?

The paper offers two qualitative analyses.

First, reconstruction examples show that the MAE can restore major retinal structures from heavily masked inputs, including vessels and optic-disc regions in CFP and layers such as the retinal nerve fibre layer in OCT.

Second, RELPROP maps,[^relprop] highlight image regions that contributed to downstream predictions. The maps emphasize haemorrhages and exudates for diabetic retinopathy, regions around the optic nerve for glaucoma, and vascular or layer-level structures for systemic tasks.[^retfound]

These findings are useful because they make the representation hypothesis plausible. They do not prove that the model has discovered the causal mechanism of disease. Saliency methods show sensitivity of a model output to image regions; they do not turn a neural-network decision into a clinical explanation.

<!-- FIGURE PLACEHOLDER 4
Use either:
(A) a carefully attributed crop of RETFound Extended Data Fig. 6 under the paper's CC BY 4.0 licence, or
(B) an original side-by-side graphic that explains the difference between reconstruction evidence, relevance evidence, and causal evidence.
Caption suggestion: "Figure 4. Reconstruction and relevance maps are supporting evidence for anatomical sensitivity, not proof of causal reasoning."
-->

## Critical reading: four reasons not to over-claim

### 1. “External” does not mean globally representative

The paper deserves credit for evaluating beyond a single internal split. However, external systemic validation moves from MEH-AlzEye to UK Biobank—two UK cohorts with different, but not globally comprehensive, participant and acquisition profiles.

Most pretraining images also come from MEH-MIDAS, a diabetic population. A representation learned from this source may encode both general retinal anatomy and characteristics correlated with diabetes care, device mix, or local referral patterns. Broader international evaluation is needed before claims about population-level generalizability become convincing.

### 2. Age is a real confounder, and the paper only partially removes it

Age affects both systemic disease incidence and retinal appearance. A model that detects age-related retinal changes can therefore appear to predict cardiovascular or neurodegenerative outcomes even if it has not learned disease-specific biology.

The authors explicitly test this concern for myocardial infarction. They hold the disease group fixed and vary the age distribution of controls. When the age gap is largest, a logistic-regression baseline using age alone reaches AUROC 0.63. As the groups become more age-matched, RETFound remains more stable than the competing models.[^retfound]

This is a valuable control experiment. It shows that the authors did not ignore age. But it does not remove every confounder: ethnicity, device, co-morbidity, socioeconomic factors, care access, and image quality can all correlate with both retinal appearance and clinical outcomes.

### 3. The SSL comparison is informative, not perfectly isolated

RETFound with MAE performs best in most of the reported SSL-comparison tasks. Yet differences among MAE, DINO, SimCLR, SwAV, and MoCo-v3 include more than the pretext objective. Architectures and hyperparameters vary, and the paper follows each method's recommended setup.[^retfound]

The appropriate interpretation is modest: **within this benchmark and implementation family, the MAE configuration was strongest**. A more definitive claim about reconstruction versus contrastive learning would require matched encoders, compute budgets, augmentations, and optimization schedules.

### 4. Retrospective accuracy is not clinical utility

Even a well-calibrated retrospective model can fail to improve care. RETFound reports lower expected calibration error than comparison models on the evaluated oculomic tasks, which is promising.[^retfound] But calibration can drift after deployment, especially when disease prevalence, devices, or referral behaviour change.

The missing evidence is prospective and workflow-based:

- Would the model change a clinician’s action?
- Would that action improve patient outcomes?
- How should uncertainty be communicated to clinicians and patients?
- What happens when one modality is unavailable or image quality is poor?
- Do errors differ systematically across demographic groups?

These questions are not objections to representation learning. They are the next scientific and clinical tests required to turn a strong retrospective model into a trustworthy tool.

## A roadmap toward clinical usefulness

Several extensions follow naturally from the analysis above.

1. **Multimodal learning.** CFP and OCT could be fused rather than modelled independently. This may combine vascular and surface information with cross-sectional retinal structure.

2. **Full-volume OCT modelling.** Replacing the middle-slice approximation with a volumetric model could recover information discarded by the current preprocessing pipeline.

3. **Richer clinical covariates.** Demographics, visual acuity, prior diagnoses, and longitudinal information may improve both prediction and confounding control—provided they are handled transparently and fairly.

4. **Geographically broader pretraining and evaluation.** A retinal foundation model should be trained and tested across device vendors, countries, healthcare systems, and patient groups.

5. **Prospective evaluation with calibration monitoring.** Performance metrics alone are insufficient. Future studies should evaluate decision impact, subgroup reliability, uncertainty handling, and post-deployment drift.

6. **Reproducibility beyond code release.** The authors release code, but key pretraining and linked clinical cohorts are controlled-access because of privacy restrictions.[^retfound] This is understandable, but it limits independent reproduction of the full training pipeline. Federated or privacy-preserving evaluation could help close that gap.

## Takeaways

RETFound is compelling because it reframes retinal AI as a representation-learning problem rather than a collection of isolated classifiers.

The paper supports three core conclusions:

1. A masked autoencoder can learn useful retinal structure from unlabelled CFP and OCT images by reconstructing missing patches.

2. Starting with generic ImageNet SSL and then specializing on retinal data improves transfer relative to the study’s alternative pretraining routes.

3. The resulting encoder can improve label efficiency and performance across ocular diagnosis, prognosis, and several systemic-risk tasks—while still facing important limits under cohort shift.

The most useful way to think about RETFound is as a strong *research foundation*: it reduces the cost of building new retinal models and provides evidence that retinal self-supervision transfers across tasks. It is not yet evidence that one universal retinal model is clinically reliable everywhere, or that retinal-image predictions can replace broader clinical assessment.

## References

[^retfound]: Zhou, Y. *et al.* (2023). *A foundation model for generalizable disease detection from retinal images*. **Nature**, 622, 156–163. [https://doi.org/10.1038/s41586-023-06555-x](https://doi.org/10.1038/s41586-023-06555-x)

[^mae]: He, K. *et al.* (2022). *Masked autoencoders are scalable vision learners*. In **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**, 16000–16009. [https://doi.org/10.1109/CVPR52688.2022.01553](https://doi.org/10.1109/CVPR52688.2022.01553)

[^vit]: Dosovitskiy, A. *et al.* (2021). *An image is worth 16×16 words: Transformers for image recognition at scale*. **International Conference on Learning Representations**. [https://openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy)

[^foundation]: Bommasani, R. *et al.* (2021). *On the opportunities and risks of foundation models*. arXiv:2108.07258. [https://doi.org/10.48550/arXiv.2108.07258](https://doi.org/10.48550/arXiv.2108.07258)

[^relprop]: Chefer, H., Gur, S., & Wolf, L. (2021). *Transformer interpretability beyond attention visualization*. In **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**, 782–791. [https://doi.org/10.1109/CVPR46437.2021.00084](https://doi.org/10.1109/CVPR46437.2021.00084)
