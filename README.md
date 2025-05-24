
# Cloud Removal from Sentinel-2 Imagery using Conditional Diffusion Models
![alt text](https://github.com/tessacannon48/cloud_removal_diffusion/blob/main/DDPM_diagram.jpg)

## About The Project

### Goal
The goal of this project is to **remove clouds from Sentinel-2 satellite imagery** using a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** with a shallow UNet architecture. The model is trained to reconstruct cloud-free scenes conditioned on their cloudy counterparts.

### Motivation
Clouds obscure satellite observations, impeding reliable analysis for remote sensing applications such as climate monitoring, land use classification, and Arctic navigation. By generating synthetic cloud-free imagery, this project helps improve data availability in frequently cloud-covered regions, with a specific focus on **North Slope, Alaska**—a relatively stable Arctic region during late spring.

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

Out of 29 initial matches, **13 pairs** were retained after manual visual inspection to ensure cloud validity and landscape consistency.

---

## Dataset Construction

**Notebook:** `/Cloud Removal Project/dataset_creation.ipynb`

1. **Patch Extraction:**
   - Cloudy images are segmented using a 256×256 sliding window.
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

### Data Limitations
- **Clear ≠ Ground Truth:** Lighting conditions vary due to clouds, so even cloud-free images don’t perfectly match the same scene under cloud cover.
- **Thin Cloud Ambiguity:** Thin clouds are hard to visually identify, particularly when downsampled into 256×256 patches.
- **Label Quality:** ESA’s cloud masks have some inaccuracies.
- **Limited Pairs:** Only 13 image pairs met the strict requirements of geographic and temporal similarity.

### Computational Constraints
- A **shallow model** was used to control training time and GPU memory.
- **Epochs and architecture depth** were limited to reduce resource usage, likely sacrificing reconstruction fidelity.

---

## Future Work

- Integrate radiometric correction or SAR data to reduce lighting inconsistencies.
- Develop better cloud mask labeling methods or integrate human-in-the-loop annotation.
- Experiment with deeper, more expressive UNet backbones if computational resources allow.
- Explore temporal modeling (e.g., clear image before/after) to improve predictions.
- Expand dataset geographically and temporally for greater generalization.

## Environmental Impact Estimation

Although this project aims to utilize AI to advance environmental technology and sustainability initiatives, it is important to recognize that there is also an environmental cost to using and building AI. Below I outline the main sources of emissions or environmental costs produced by this project, from the collection of data to the model development process. 

- Environmental Costs of Data Acquisition (Sentinel-2 Satellite)
  - Satellites Used: ESA Sentinel-2
- Model Training Impact
  - Individual model training time: 10k model: 30 minutes, 20k model: 2 hours, full model: 7.5 hours
  - Cumulative estimated training time (including time for debugging + retraining): 24 hours of training
  - Hardware used: NVIDIA GeForce RTX 4090 GPUs + Apple M3 10-Core GPU
  - Emissions Estimate:
  - Water Usage Estimate: 
- Generative AI Usage
  - Generative Large Language Models (ChatGPT) were used as a resource for various tasks in this project, such as for debugging and cleaning code. I estimate that I asked about 30 queries to ChatGPT, generating ____ of CO2, according to _____. 


## Acknowledgements

- This project was completed as part of an open-ended final coursework for GEOL0069: Artificial Intelligence for Earth Observation taught by Dr. Michel Tsamados
- Some ESA Query functions provided by Dr. Michel Tsamados, Weibin Chen, Connor Nelson
- Satellite imagery provided by [Copernicus Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) via ESA Copernicus Data Space Ecosystem.
- Diffusion model structure inspired by [Ho et al., 2020](https://arxiv.org/abs/2006.11239).
