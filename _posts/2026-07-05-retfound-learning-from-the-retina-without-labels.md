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

## From task-specific classifiers to retinal foundation models

## Masked autoencoders: learning by reconstructing hidden content

## RETFound's sequential pretraining strategy

## From pretrained features to clinical predictions

## How was RETFound evaluated?

## What did the experiments show?

## What did RETFound actually learn?

## Limitations and future directions

## Takeaways

## References
