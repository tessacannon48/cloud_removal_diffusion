
# Cloud Removal from Sentinel-2 Imagery using Conditional Diffusion Models
![alt text](https://github.com/tessacannon48/cloud_removal_diffusion/blob/main/DDPM_diagram.jpg)

## About The Project

### YouTube Tutorial

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/0CCpH0Z3Mrw/0.jpg)](https://www.youtube.com/watch?v=0CCpH0Z3Mrw)

### Goal
The goal of this project is to **remove clouds from Sentinel-2 satellite imagery** using a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** with a shallow UNet architecture. The model is trained to reconstruct cloud-free scenes conditioned on their cloudy counterparts.

### Motivation
Clouds obscure satellite observations, impeding reliable analysis for remote sensing applications such as climate monitoring, land use classification, and Arctic navigation. By generating synthetic cloud-free imagery, this project helps improve data availability in frequently cloud-covered regions, with a specific focus on **North Slope, Alaska**—an Arctic region chosen due to it having relatively stable ice cover during the late spring.

### Background

Recent advances in generative machine learning have made it possible to reconstruct cloud-free satellite imagery from cloud-contaminated inputs, with diffusion models emerging as a particularly promising approach. In particular, DiffCR (Zou et al., 2023) introduced a conditional denoising diffusion probabilistic model (DDPM) that learns to generate high-fidelity, cloud-free Sentinel-2 images conditioned on their cloudy counterparts [[1]](#1). DiffCR builds on earlier diffusion work like DDPM-CR by using a lightweight U-Net architecture and a novel Time-Condition Fusion Block for efficient inference, achieving state-of-the-art results while drastically reducing computational demands. These models outperform traditional techniques by better preserving spatial detail and color consistency, even under thick cloud cover.

Prior to diffusion models, Generative Adversarial Networks (GANs) were the dominant method for cloud removal, including paired-image GANs, CycleGANs for unpaired training, and spatiotemporal GANs leveraging multi-temporal sequences. Some models also used multi-modal data, combining Sentinel-2 optical imagery with Sentinel-1 SAR to infer obscured terrain. Other approaches include variational autoencoders (VAEs) and temporal convolutional networks, with varying degrees of success. Overall, the field has shifted from heuristic or interpolation-based methods toward data-driven restoration techniques, with diffusion models now offering the best balance of accuracy and flexibility for cloud removal—especially when applied to Sentinel-2 imagery.

While the techniques used in this project are not novel, to the best of my knowledge, diffusion models have not yet been applied to remove clouds from satellite imagery over Arctic sea ice. Furthermore, the creation of a custom dataset composed of Sentinel-2 imagery specifically curated for this task adds an additional element of novelty to the project.

---

## Getting Started

### Installation
To set up the environment:

```bash
git clone https://github.com/tessacannon48/cloud_removal_diffusion.git
cd cloud_removal_diffusion
pip install -r requirements.txt
```

### Repository Structure

```
Cloud Removal Project
├── data_retrieval.ipynb          # Queries and downloads Sentinel-2 data
├── dataset_creation.ipynb        # Extracts and filters training patches
├── model_development.ipynb       # Defines and trains the DDPM model
├── temp/
│   ├── cloudy_patches/           # Cloudy image patches
│   ├── clear_patches/            # Clear image patches
│   └── mask_patches/             # Cloud masks
├── cloudless_data/               # Zipped clear Sentinel-2 scenes
├── cloudy_data/                  # Zipped cloudy Sentinel-2 scenes
└── requirements.txt              # Python dependencies
```

---

## Data Source

**Notebook:** `/Cloud Removal Project/data_retrieval.ipynb`  
**Provider:** [ESA Copernicus Data Space](https://dataspace.copernicus.eu/)

- **Region:** North Slope, Alaska  
- **Bounding Polygon:**  
  `(-162.0 68.5, -162.0 71.5, -140.0 71.5, -140.0 68.5, -162.0 68.5)`
- **Time Window:**
  - **Clear Images:** March 1–30, 2024 (≤1% cloud coverage)
  - **Cloudy Images:** February 1–April 30, 2024 (5–50% cloud coverage)

Image pairs are matched by:
- Same geographic area
- Temporal proximity (±30 days)
- Preference for closest match if multiple options exist

Out of 29 initial matches, **13 pairs** were retained after manual visual inspection to ensure cloud mask validity and landscape consistency.

---

## Dataset Construction

**Notebook:** `/Cloud Removal Project/dataset_creation.ipynb`

1. **Patch Extraction:**
   - Cloudy images are segmented using a 256×256 sliding window.
   - Only 4 spectral bands are kept: RBG + NIR (10m)
   - Selected only those patches where **cloud coverage is 5–30%**, ensuring good coverage and variability for learning.

2. **Output:**
   - 37,661 patch pairs, each with:
     - A cloudy image patch
     - A corresponding clear image patch
     - A cloud mask patch
   - Stored in:
     - `temp/cloudy_patches/`
     - `temp/clear_patches/`
     - `temp/mask_patches/`

---

## Modelling

**Notebook:** `/Cloud Removal Project/model_development.ipynb`  
**Model:** Conditional Denoising Diffusion Probabilistic Model (DDPM)  
**Architecture:** Lightweight UNet

### Architecture Details
- **Conditioning Input:** Cloudy image patch
- **Target Output:** Cloud-free image patch
- **Downsampling:** 2 layers via MaxPooling
- **Upsampling:** 2 layers via Transposed Convolution
- **Blocks:** Double convolution layers (GELU + GroupNorm)
- **Timestep Encoding:** Sinusoidal embedding
- **Loss Function:** MSE weighted by the cloud mask (to focus reconstruction on cloud-covered areas)

### Training Configuration
- Epochs: 5  
- Batch Size: 8  
- Diffusion Timesteps: 1000  
- Learning Rate: `1e-4`  
- Evaluation Metrics (on samples):  
  - Visual inspection  
  - SSIM  
  - PSNR

---


## Limitations

While there is evidence that the full model is able to generate some accurate structure from the clear images prompted by the cloudy image, the images being generated are currently blanketed in a layer of noise. Therefore, additional training and tuning is required to achieve accurate reconstructions. Please see the attached video above for an in-depth discussion of the modeling results. Below I outline several limitations which are contributing to the inaccuracy of the reconstructions: 

### Data Limitations
- **Clear ≠ Ground Truth:** Lighting conditions vary due to clouds, so even cloud-free images don’t perfectly match the same scene under cloud cover.
- **Thin Cloud Ambiguity:** Thin clouds are hard to visually identify, particularly when downsampled into 256×256 patches.
- **Label Quality:** ESA’s cloud masks have some inaccuracies.
- **Limited Pairs:** Only 13 image pairs met the strict requirements of geographic and temporal similarity.

### Computational Constraints
- A **shallow model** was used to control training time and GPU memory.
- **Epochs and architecture depth** were limited to reduce computational complexity, likely sacrificing reconstruction accuracy. 
---

## Future Work

- Fine-tune the model architecture and set of hyperparameters. Experiment with deeper, more expressive UNet backbones.
- Perform ablation studies and model comparisons to effectively evaluate DDPMs.
- Integrate radiometric correction or SAR data to reduce lighting inconsistencies.
- Develop better cloud mask labeling methods or integrate human-in-the-loop annotation.
- Expand dataset geographically and temporally for greater generalization.

## Environmental Impact Estimation

The techniques developed in this project could be used to enhance a variety of sustainable initiatives involving Earth observation. Removing clouds from satellite imagery using diffusion models significantly enhances the quality and usability of Earth observation data, which is crucial for environmental monitoring and management. Cloud cover can obscure critical information in satellite images, hindering accurate analysis of land use, vegetation health, and water resources. By employing advanced cloud removal techniques, researchers can obtain clearer images, leading to more precise assessments of environmental conditions. This improved clarity aids in tracking deforestation, monitoring agricultural practices, and managing natural disasters more effectively. Consequently, the application of diffusion models for cloud removal contributes to more informed decision-making in environmental conservation and sustainable resource management.

Although this project aims to utilize AI to advance environmental technology and sustainability initiatives, it is important to recognize that there is also an environmental cost to using and building AI. Below I outline the main sources of emissions or environmental costs produced by this project, from the collection of data to the model development process. 

**Satellite Data Acquisition (Sentinel-2):** The Sentinel-2 mission carries a notable upfront carbon cost. Launching a single Sentinel-2 Earth-observation satellite on a conventional rocket produces a significant amount of CO2 emissions. For example, SpaceX’s Falcon 9 rocket emits 425 tonnes of carbon per launch. However, it is important to note that this impact is one-off, meaning it does not continue emitting carbon after the initial launch. The fact that these emissions are one-off makes satellite monitoring much more sustainable in the long-term when compared to alternative methods of earth observation, such as operating helicopters to monitor vegetation [[2]](#2). Additionally, for this project alone, even though the use of Sentinel-2 images has a substantial embedded carbon impact from the satellite’s deployment, the marginal emissions of downloading a few dozen images are negligible by comparison.

**Model Training (24 h on GPUs):** The project’s AI model training phase (approximately 24 hours split between an NVIDIA RTX 4090 and Apple M3 10-core GPU) also incurred a measurable environmental impact. According to research on carbon emissions from deep learning conducted at the University of Massachusetts Amherst, the Transformer base model with 65 million parameters—the smallest model evaluated by the researchers—emitted 0.01 tonnes of CO₂ after 12 hours of training across 8 NVIDIA P100 GPUs [[3]](#3). In contrast, my model was significantly smaller, with only 3,452,868 parameters, and was trained for a cumulative total of 24 hours on a single GPU at a time. Although it is difficult to make a direct comparison with this study, it is reasonable to assume that my model consumed significantly less energy than the Transformer base model. Therefore, while the environmental impact is worth acknowledging, the carbon emissions from my model are likely to be substantially lower than the UK’s average per-person annual emission of 10 tonnes of CO2 [[4]](#4). 

**Generative AI Usage (ChatGPT Queries):** Lastly, the interactive use of a generative AI (approximately 30 ChatGPT queries) contributed a relatively minor carbon emission in this project. Running large language models is computationally intensive, but each text query is short-lived; estimates put the energy for one ChatGPT query on the order of only a few watt-hours, corresponding to roughly 2–5 g of CO2 per query [[5]](#5). Although generative AI requests use several times more energy than a standard Google search (≈0.2 g CO2 per query), the ~30 queries would have emitted approximately 100 g of CO2, a trivial amount in the context of the overall project. Altogether, the dominant environmental impacts of the project stem from the satellite infrastructure and model training phases, whereas the direct emissions from a handful of AI queries are comparatively insignificant. 

The environmental costs of employing this project could be reduced through more careful debugging and construction prior to full training to reduce time spent on retraining models, as well as reduced reliance on generative AI for debugging. To monitor and estimate the carbon emissions from the code, Python packages like CodeCarbon (found at https://github.com/mlco2/codecarbon) could be employed prior to development in future iterations of this project. Overall, however, the environmental costs of developing this project alone are minimal based on the impact estimation above. 

## Acknowledgements

- This project was completed as part of an open-ended final coursework for GEOL0069: Artificial Intelligence for Earth Observation taught by Dr. Michel Tsamados
- Some ESA Query functions used in data retrieval were provided by Dr. Michel Tsamados, Weibin Chen, Connor Nelson
- Satellite imagery provided by [Copernicus Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) via ESA Copernicus Data Space Ecosystem.

## References
<a id="1">[1]</a> 
https://ieeexplore.ieee.org/document/10436560

<a id="2">[2]</a> 
https://www.live-eo.com/article/how-green-is-satellite-monitoring-lets-do-the-math

<a id="3">[3]</a> 
https://arxiv.org/pdf/1906.02243

<a id="4">[4]</a> 
https://www.carbonindependent.org/23.html

<a id="5">[5]</a> 
https://smartly.ai/blog/the-carbon-footprint-of-chatgpt-how-much-co2-does-a-query-generate



