# FinalYearProjectVault

A centralized repository for projects, reports, PPTs, and code submitted by students working under the guidance of Shounak Chakraborty.

## Overview

FinalYearProjectVault serves as a comprehensive management system for final year student projects. This repository maintains organized documentation, project files, presentations, and source code, providing a single source of truth for all submissions and project artifacts.

## Purpose

This repository enables:
- Centralized storage of student project deliverables
- Easy access and tracking of project progress
- Organization of reports, presentations, and code in a structured manner
- Collaboration between students and faculty advisors
- Version control and historical tracking of project development

## Repository Structure

The repository is organized to accommodate multiple student projects with the following typical structure:

```
FinalYearProjectVault/
├── README.md                 # This file
├── [Project folders]/        # Individual project directories
│   ├── reports/             # Project reports and documentation
│   ├── presentations/       # PowerPoint presentations and slides
│   ├── code/                # Source code and implementation files
│   └── [other assets]/      # Additional project materials
```

## Getting Started

### Prerequisites

- Git installed for cloning and updating the repository
- Access credentials if the repository requires authentication
- A text editor or IDE for viewing project documentation and code

### Cloning the Repository

```bash
git clone <repository-url>
cd FinalYearProjectVault
```

## Submission Guidelines

When submitting a project to this vault:

1. Create a new folder with your project name
2. Organize submissions into logical subdirectories (reports, presentations, code, etc.)
3. Include a project-specific README describing your work
4. Ensure all files are properly named and documented
5. Update the main README with your project details

## Faculty Advisor

**Shounak Chakraborty** - Project Guide

## Contributing

Students should follow Git best practices when contributing:
- Create feature branches for new work
- Write clear commit messages
- Update documentation as changes are made
- Ensure all submissions follow the repository structure guidelines

## Notes

- This repository is intended for academic project management
- All submissions should maintain confidentiality and academic integrity standards
- Regular backups of critical project files are recommended

## License

Please check with the repository administrator or institution for licensing details.

## Projects

### 2026

#### Joint Super-Resolution and Multi-Label Classification for Remote Sensing Imagery
**Student:** Shreya Pragna  
**Directory:** `Final-Year-Projects/2026/sr_mlc_Shreya_Pragna/`

**Overview:**  
This project presents a joint deep learning framework that integrates Satellite Image Super-Resolution (SR) and Multi-Label Classification (MLC) for remote sensing imagery. The framework enhances spatial resolution of low-quality satellite images while simultaneously performing land-cover classification, enabling accurate semantic understanding under severe resolution constraints.

**Key Components:**
- **Dataset:** FLAIR-2 with Sentinel-2 (10m resolution) paired with high-resolution aerial imagery (0.2m)
- **SR Module:** Feature-space super-resolution using Conditional Variational Autoencoder (CVAE) with multi-objective losses (perceptual, edge-aware, SSIM)
- **MLC Module:** ResNet50 backbone with GeM pooling, Graph Neural Networks (GNN) for spatial and label reasoning
- **Training Strategy:** Progressive three-stage approach (SR pretraining → MLC pretraining → joint fine-tuning)
- **Evaluation Metrics:** PSNR, SSIM for SR; Precision, Recall, F1-score, mAP for MLC

**Submission Contents:**
- `report.pdf` – Comprehensive technical report
- `final_presentation.pdf` – Project presentation slides
- `data-preparation.ipynb` – Data preprocessing and dataset preparation pipeline
- `joint-sr-mlc-models-2025.ipynb` – Complete model implementation and training code
- `figures/` – Architecture diagrams and sample results
- `README.md` – Detailed project documentation
