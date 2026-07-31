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

> **Scope of this post.** This is not a section-by-section summary of Zhou et al. Instead, it reads RETFound as a representation-learning experiment: what is the learning problem, what does each comparison isolate, and how far do the reported results justify the clinical claims?

## The question behind RETFound

Retinal imaging is an attractive setting for medical machine learning because eye-care systems routinely collect large numbers of colour fundus photographs (CFP) and optical coherence tomography (OCT) scans. The images are abundant, but the labels needed for conventional supervised learning are much scarcer. A diabetic-retinopathy grade, a glaucoma assessment, or a future cardiovascular outcome may require specialist review, longitudinal records, or both.

This creates a familiar problem. A task-specific classifier can perform well when it is trained on a well-labelled dataset collected for exactly that task. It is less obvious how to reuse the same model for another disease, another imaging device, or another hospital.

RETFound asks whether the expensive clinical supervision can be moved to the end of the pipeline. The model first learns a broad representation of retinal structure from unlabelled images. Only afterwards is it fine-tuned with labelled data for specific clinical tasks.

The paper therefore tests a stronger hypothesis than simply "self-supervision helps". Its central claim is that **sequential self-supervised pretraining—first on natural images, then on retinal images—produces representations that transfer better than the tested alternatives**.

The downstream tasks fall into three groups:

1. **Diagnosis:** detecting disease that is already present in the image;
2. **Prognosis:** predicting whether a fellow eye will convert to wet age-related macular degeneration (wet AMD) within one year;
3. **Oculomics:** estimating the three-year incidence of systemic conditions from retinal images.

The third setting needs careful wording. A retinal scan does not directly diagnose heart failure, myocardial infarction, stroke, or Parkinson's disease. RETFound outputs a statistical risk estimate learned from associations in retrospective data. It does not establish a causal mechanism, and it should not be interpreted as a stand-alone clinical decision-maker.

<figure>
  <img src="/images/retfound/cfp-oct-comparison.png" alt="CFP and OCT retinal imaging modalities" style="width:100%;">
  <figcaption><strong>Figure 1.</strong> CFP provides a surface view of the retina, while OCT provides a cross-sectional view of retinal layers.</figcaption>
</figure>


## Retinal images are not one kind of data

RETFound uses two imaging modalities with different visual structure.

### Colour fundus photography: a surface view

A colour fundus photograph is a two-dimensional colour image of the retinal surface. It shows blood vessels, the macula, the optic disc, and visible lesions such as haemorrhages or lipid exudates. This makes CFP useful for diseases where colour, vascular geometry, or surface-level pathology are informative.

### Optical coherence tomography: a cross-sectional view

Optical coherence tomography provides a cross-sectional image through the retina. It reveals the layered microstructure of the tissue and can show changes in thickness, fluid, or layer integrity that a surface photograph may miss.

A simple mental model is useful: **CFP asks what the retina looks like from above; OCT asks what it looks like in cross-section.** The two modalities are complementary, but RETFound does not fuse them. The authors train one model for CFP and another model for OCT.

That separation is methodologically sensible because the two modalities have different appearance and disease cues. It is also a limitation. A future multimodal model could potentially combine vascular information from CFP with layer-level information from OCT.

### What reaches the network?

The preprocessing choices matter because they define what information the model can use.

For CFP, the authors use AutoMorph to remove the background and retain the retinal field. For OCT, they extract the middle slice from the scan rather than modelling the full volumetric acquisition. Both modalities are resized to 256 × 256 pixels, then augmented with random crops resized to 224 × 224, random horizontal flips, and normalization [1].

The OCT choice is important. Taking one central B-scan makes a large-scale Transformer pipeline tractable, but it discards information away from that slice. The OCT results should therefore be read as results for **single-slice OCT representation learning**, not as evidence that the full 3D OCT volume has been exploited.

## RETFound as a machine-learning problem

A useful way to understand RETFound is to separate its two objectives. During pretraining, the model solves a self-supervised image-reconstruction problem. During fine-tuning, it solves a supervised clinical prediction problem.

| Stage | Input | Output / target | Learning objective |
|---|---|---|---|
| **Masked-autoencoder pretraining** | A retinal image with many patches hidden | Reconstruction of the missing patches | Minimize reconstruction error on masked content |
| **Task-specific fine-tuning** | A complete CFP or OCT image | Disease class or event probability | Minimize supervised classification loss with label smoothing |

### Stage 1: reconstructing masked retinal content

Let a preprocessed image be denoted by $x$. After the 224 × 224 crop, RETFound divides it into 16 × 16 patches, producing $N = 14 \times 14 = 196$ visual tokens. Let $M$ be the masked patch indices and $V$ the visible patch indices.

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

where $\ell$ is a reconstruction-error term evaluated only on the hidden patches. The paper describes the objective as reconstruction error; the equation above is a compact way to express the masked-patch objective rather than an additional modelling assumption [1,2].

This is the "self" in self-supervised learning: the image supplies its own training target. No disease label is required during pretraining.

### Stage 2: predicting a labelled clinical outcome

For a downstream task, the MAE decoder is discarded. The retained encoder maps a complete image $x$ to a representation $h$, and a multilayer perceptron head $g_{\psi}$ maps that representation to class probabilities or event probabilities:

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

For binary prognosis and incidence tasks, the same logic reduces to estimating the probability of the event of interest. The paper fine-tunes the downstream models for 50 epochs and saves the checkpoint with the highest validation AUROC [1].

The two-stage distinction is the core of the method:

- Pretraining asks: **What visual structure is needed to complete a retinal image?**
- Fine-tuning asks: **Which parts of that learned representation help predict this clinical outcome?**

## Learning anatomy from missing patches

RETFound uses a masked autoencoder (MAE), a self-supervised architecture introduced for image representation learning by He et al. [2]. The model deliberately hides most of its input, processes only the visible part with a large encoder, and reconstructs the missing content with a lighter decoder.

<figure>
  <img src="/images/retfound/mae-pretraining-pipeline.png" alt="Self-supervised masked-autoencoder pretraining pipeline for RETFound" style="width:100%;">
  <figcaption><strong>Figure 2.</strong> RETFound learns from unlabelled retinal images by masking many patches and reconstructing the missing content.</figcaption>
</figure>


### Why mask so aggressively?

RETFound masks 75% of CFP patches and 85% of OCT patches [1]. At first, this may sound counterintuitive. Why make the input so incomplete?

The reason is to avoid easy local shortcuts. If only a small region were removed, the decoder might fill it in by interpolating nearby texture. With most of the image hidden, the model has to use broader retinal regularities: vessel continuity, typical optic-disc position, fundus geometry, or the layered organization of OCT.

This does **not** mean that the model learns medical reasoning in a human sense. It means that good reconstruction becomes difficult without representing recurring anatomical context. Whether that context is useful for disease prediction is then tested empirically in the downstream tasks.

### Architecture and training scale

The encoder is ViT-Large, with 24 blocks and 1,024 features per token [3]. Reconstruction is handled by a smaller eight-block decoder with 512-dimensional token features. The retinal pretraining stage uses 800 epochs, a batch size of 1,792, and eight NVIDIA A100 GPUs; the authors report about fourteen days of pretraining time [1].

| Component | RETFound configuration |
|---|---|
| Encoder | ViT-Large; 24 blocks; 1,024 features per token |
| Decoder | Smaller ViT; 8 blocks; 512 features per token |
| Patch size | 16 × 16 pixels |
| CFP masking ratio | 75% |
| OCT masking ratio | 85% |
| Retinal SSL corpus | about 900,000 CFP images + about 700,000 OCT images |
| Retinal SSL compute | 8 × A100 GPUs, about 14 days |

This scale is both a strength and a caveat. It makes a rich retinal representation plausible, but it also means that training a comparable foundation model from scratch is not equally accessible to every clinical research group.

## Why the pretraining order is itself an experiment

The most interesting part of RETFound is not only its architecture. It is the four-way comparison used to ask **where useful representations come from**.

<figure>
  <img src="/images/retfound/pretraining-strategies.png" alt="Four pretraining strategies compared in RETFound" style="width:100%;">
  <figcaption><strong>Figure 3.</strong> The four comparison strategies test whether supervised ImageNet transfer, generic self-supervision, retinal self-supervision, or their sequential combination gives the strongest representation. Adapted from Zhou et al. (2023).</figcaption>
</figure>


| Model | Pretraining route | Main question addressed |
|---|---|---|
| **SL-ImageNet** | Supervised learning on ImageNet-21k | How strong is conventional transfer from labelled natural images? |
| **SSL-ImageNet** | Self-supervised learning on ImageNet-1k | How far do generic SSL image representations transfer without retinal specialization? |
| **SSL-Retinal** | SSL on retinal images from random initialization | Is retinal-domain SSL sufficient on its own? |
| **RETFound** | SSL on ImageNet-1k, then SSL on retinal images | Does generic SSL initialization add value before retinal specialization? |

The cleanest comparison is **SSL-Retinal versus RETFound**. Both use retinal self-supervised learning, but RETFound begins from an ImageNet-SSL representation. If RETFound performs better, this supports the idea that generic visual structure helps the later retinal stage.

The other comparisons require more care. **SL-ImageNet versus SSL-ImageNet** changes both the supervision type and the source dataset: SL-ImageNet uses ImageNet-21k with labelled images, while SSL-ImageNet uses ImageNet-1k with self-supervised learning. It is therefore not a perfectly controlled "supervised versus self-supervised" ablation.

The comparison with other SSL methods is also informative but not perfectly isolated. The paper replaces MAE with SimCLR, SwAV, MoCo-v3, and DINO in the RETFound framework, but the recommended architectures and hyperparameters differ across methods [1]. A careful conclusion is therefore: **within the tested configurations, MAE is the strongest overall strategy in this study**. The experiment does not prove that reconstruction-based SSL is universally superior to contrastive or self-distillation methods.

## Evaluation: which claims are actually tested?

RETFound is evaluated on several types of experiments. Each supports a different claim, and each has limits.

<figure>
  <img src="/images/retfound/evaluation-task-families.png" alt="RETFound task families and internal or external evaluation settings" style="width:100%;">
  <figcaption><strong>Figure 4.</strong> RETFound is evaluated across ocular diagnosis, wet-AMD prognosis, and systemic risk prediction, with different internal and external validation settings.</figcaption>
</figure>


| Claim | Experimental test | What the test can support | What it cannot establish |
|---|---|---|---|
| Better ocular transfer | Cross-dataset diabetic-retinopathy evaluation | Robustness across selected retinal datasets | Universal robustness to every clinic or country |
| Better future-risk prediction | One-year wet-AMD and three-year systemic-incidence tasks | Predictive association within defined cohorts | Causality or clinical benefit |
| Better label efficiency | Fine-tuning with 10–100% of labels | Lower label demand for selected tasks | That labels are unnecessary |
| Better calibration | Reliability diagrams and expected calibration error | Better probability agreement in retrospective test data | Safe deployment without monitoring |
| Clinically meaningful representations | Reconstructions and RELPROP maps | Plausible anatomical or pathological focus | A complete explanation of the model's reasoning |

### Data sources and task families

The retinal pretraining corpus contains about 1.6 million images: roughly 900,000 CFP images and 700,000 OCT images. The dominant source is MEH-MIDAS, a retrospective Moorfields collection of imaging records from people with diabetes. EyePACS contributes additional CFP data, and a public OCT dataset contributes additional OCT data [1].

For ocular diagnosis, the authors use public datasets covering diabetic retinopathy, glaucoma, and multi-category retinal disease. For wet-AMD prognosis and systemic prediction, they use the MEH-AlzEye record-linkage cohort. The systemic external evaluation transfers models trained on MEH-AlzEye to UK Biobank [1].

This distinction matters. Some ocular diagnosis experiments test transfer across public datasets. The systemic external evaluation is a real cohort shift, but both source and target are UK-based. It should not be equated with worldwide validation.

### Metrics and uncertainty

The paper reports AUROC and AUPR.

- **AUROC** summarizes how well the model ranks positive cases above negative cases across thresholds.
- **AUPR** is especially useful when positive cases are rare, because it focuses on the precision-recall trade-off.

Each task is trained with five random seeds. The reported confidence intervals are calculated from variation across these five training runs, and statistical tests compare RETFound with the strongest competing model [1].

This is useful for assessing sensitivity to optimization randomness. It is not the same as uncertainty across independent hospitals or healthcare systems. Seed-based intervals can be narrow even when the larger uncertainty is the shift from one clinical environment to another.

## What the experiments support—and where the evidence stops

### Ocular diagnosis: the strongest transfer evidence

The cross-dataset diabetic-retinopathy experiments provide some of the clearest evidence that RETFound learns transferable features. A model fine-tuned on one dataset is evaluated on another rather than only on a held-out split from the same source.

For example, when fine-tuned on APTOS-2019, RETFound achieved AUROC 0.822 on IDRiD and 0.738 on MESSIDOR-2. The paper reports that RETFound ranked first in all cross-evaluations among the compared models [1].

This is stronger than a random train-test split because acquisition conditions, label protocols, and population characteristics can differ between datasets. Still, it is transfer across a limited set of retinal benchmarks, not proof that the model will work unchanged in every clinic.

### Prognosis: predicting wet-AMD conversion

For one-year prediction of wet-AMD conversion in the fellow eye, RETFound reaches AUROC 0.862 with CFP and 0.799 with OCT [1].

This task is clinically interesting because it asks the model to predict a future event rather than recognize a disease that is already present. The "fellow eye" framing matters: the input comes from the eye that has not yet converted, so the model is searching for risk-associated patterns before the target event occurs.

The result is encouraging, but it remains an internal evaluation on the AlzEye cohort. Prospective validation would be needed to know whether the prediction changes surveillance, treatment timing, or patient outcomes.

### Oculomics: useful signal, difficult interpretation

The systemic tasks are the most ambitious and the easiest to overstate. In internal CFP evaluation, RETFound reports AUROCs of 0.737 for myocardial infarction, 0.794 for heart failure, 0.754 for ischaemic stroke, and 0.669 for Parkinson's disease [1].

These outcomes show that the learned retinal representation improves relative ranking over the baselines in the paper. They do **not** show that a retinal image alone is sufficient for high-stakes systemic diagnosis.

The external results reinforce this caution. Performance decreases after transfer to UK Biobank. For external ischaemic-stroke prediction, RETFound and SSL-Retinal perform similarly rather than showing a clear advantage for sequential pretraining. The paper attributes part of the challenge to shifts in demographics, ethnicity, and imaging devices, and reports a notable internal-to-external AUROC drop for stroke prediction [1].

The balanced conclusion is: retinal images appear to carry predictive information about systemic risk, and retinal self-supervision helps access that information. The paper does not demonstrate a causal pathway from retinal appearance to systemic disease, nor a ready-to-deploy screening tool.

### Label efficiency: a practically important result

RETFound's most immediately useful contribution may be the label-efficiency analysis. The authors fine-tune models using different fractions of the available labels and find that RETFound outperforms the comparison models with only 10% of the labelled data for heart-failure and myocardial-infarction prediction [1].

This does not mean that 90% of labels can always be discarded. The result is task-specific and depends on the target performance level. What it shows is that a pretrained retinal encoder can shift part of the learning burden away from downstream clinical annotation.

The adaptation-efficiency experiment points in the same direction. For myocardial-infarction prediction, RETFound reaches the selected validation checkpoint substantially earlier, corresponding to an estimated 80% reduction in training time relative to the strongest comparison model. For diabetic retinopathy on MESSIDOR-2, the corresponding estimate is 46% [1]. These are convergence-time estimates under the reported setup, not a universal promise of identical wall-clock savings.

### Does the model attend to medically meaningful structures?

The paper offers two qualitative analyses.

First, reconstruction examples show that the MAE can restore major retinal structures from heavily masked inputs, including vessels and optic-disc regions in CFP and retinal layers in OCT.

Second, RELPROP maps highlight image regions that contributed to downstream predictions [4]. The maps emphasize haemorrhages and exudates for diabetic retinopathy, regions around the optic nerve for glaucoma, and vascular or layer-level structures for systemic tasks [1].

These findings make the representation hypothesis plausible. They do not prove that the model has discovered a causal mechanism of disease. Saliency methods show sensitivity of a model output to image regions; they do not turn a neural-network decision into a complete clinical explanation.

## Critical reading: four reasons not to over-claim

### 1. "External" does not mean globally representative

The paper deserves credit for evaluating beyond a single internal split. However, external systemic validation moves from MEH-AlzEye to UK Biobank: two UK cohorts with different, but not globally comprehensive, participant and acquisition profiles.

Most pretraining images also come from MEH-MIDAS, a diabetic population. A representation learned from this source may encode both general retinal anatomy and characteristics correlated with diabetes care, device mix, or local referral patterns. Broader international evaluation is needed before claims about population-level generalizability become convincing.

### 2. Age is a real confounder, and the paper only partially removes it

Age affects both systemic disease incidence and retinal appearance. A model that detects age-related retinal changes can therefore appear to predict cardiovascular or neurodegenerative outcomes even if it has not learned disease-specific biology.

The authors explicitly test this concern for myocardial infarction. They hold the disease group fixed and vary the age distribution of controls. When the age gap is largest, an age-only logistic-regression baseline reaches AUROC 0.63. As the groups become more age-matched, RETFound remains more stable than the competing models [1].

This is a valuable control experiment. It shows that the authors did not ignore age. But it does not remove every confounder: ethnicity, device, co-morbidity, socioeconomic factors, healthcare access, and image quality can all correlate with both retinal appearance and clinical outcomes.

### 3. The SSL comparison is informative, not perfectly isolated

RETFound with MAE performs best in most of the reported SSL-comparison tasks. Yet differences among MAE, DINO, SimCLR, SwAV, and MoCo-v3 include more than the pretext objective. Architectures and hyperparameters vary, and the paper follows each method's recommended setup [1].

The appropriate interpretation is modest: **within this benchmark and implementation family, the MAE configuration was strongest**. A more definitive claim about reconstruction versus contrastive learning would require matched encoders, compute budgets, augmentations, and optimization schedules.

### 4. Retrospective accuracy is not clinical utility

Even a well-calibrated retrospective model can fail to improve care. RETFound reports lower expected calibration error than comparison models on the evaluated oculomic tasks, which is promising [1]. But calibration can drift after deployment, especially when disease prevalence, devices, or referral behaviour change.

The missing evidence is prospective and workflow-based:

- Would the model change a clinician's action?
- Would that action improve patient outcomes?
- How should uncertainty be communicated to clinicians and patients?
- What happens when one modality is unavailable or image quality is poor?
- Do errors differ systematically across demographic groups?

These questions are not objections to representation learning. They are the next scientific and clinical tests required to turn a strong retrospective model into a trustworthy tool.

## A roadmap toward clinical usefulness

Several extensions follow naturally from the analysis above.

1. **Multimodal learning.** CFP and OCT could be fused rather than modelled independently. This may combine vascular and surface information with cross-sectional retinal structure.

2. **Full-volume OCT modelling.** Replacing the middle-slice approximation with a volumetric model could recover information discarded by the current preprocessing pipeline.

3. **Richer clinical covariates.** Demographics, visual acuity, prior diagnoses, and longitudinal information may improve both prediction and confounding control, provided they are handled transparently and fairly.

4. **Geographically broader pretraining and evaluation.** A retinal foundation model should be trained and tested across device vendors, countries, healthcare systems, and patient groups.

5. **Prospective evaluation with calibration monitoring.** Performance metrics alone are insufficient. Future studies should evaluate decision impact, subgroup reliability, uncertainty handling, and post-deployment drift.

6. **Reproducibility beyond code release.** The authors release code, but key pretraining and linked clinical cohorts are controlled-access because of privacy restrictions [1]. This is understandable, but it limits independent reproduction of the full training pipeline. Federated or privacy-preserving evaluation could help close that gap.

## Takeaways

RETFound is compelling because it reframes retinal AI as a representation-learning problem rather than a collection of isolated classifiers.

The paper supports three main conclusions.

First, a masked autoencoder can learn useful retinal structure from unlabelled CFP and OCT images by reconstructing missing patches.

Second, starting with generic ImageNet self-supervision and then specializing on retinal data improves transfer relative to the study's alternative pretraining routes.

Third, the resulting encoder can improve performance and label efficiency across ocular diagnosis, wet-AMD prognosis, and several systemic-risk tasks, while still facing important limits under cohort shift.

The most useful way to think about RETFound is as a strong **research foundation**. It reduces the cost of building new retinal models and provides evidence that retinal self-supervision transfers across tasks. It is not evidence that one universal retinal model is clinically reliable everywhere, or that retinal-image predictions can replace broader clinical assessment.

## References

[1] Zhou, Y. et al. (2023). *A foundation model for generalizable disease detection from retinal images*. **Nature**, 622, 156–163. https://doi.org/10.1038/s41586-023-06555-x

[2] He, K. et al. (2022). *Masked autoencoders are scalable vision learners*. In **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**, 16000–16009. https://doi.org/10.1109/CVPR52688.2022.01553

[3] Dosovitskiy, A. et al. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale*. **International Conference on Learning Representations**. https://openreview.net/forum?id=YicbFdNTTy

[4] Chefer, H., Gur, S., & Wolf, L. (2021). *Transformer interpretability beyond attention visualization*. In **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**, 782–791. https://doi.org/10.1109/CVPR46437.2021.00084

[5] Bommasani, R. et al. (2021). *On the opportunities and risks of foundation models*. arXiv:2108.07258. https://doi.org/10.48550/arXiv.2108.07258
