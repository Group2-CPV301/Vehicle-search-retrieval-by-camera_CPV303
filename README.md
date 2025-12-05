# 🚗 Advanced Vehicle Tracking & Retrieval System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green)
![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-orange)
![CLIP](https://img.shields.io/badge/AI-OpenAI%20CLIP-purple)

> **An intelligent traffic monitoring system capable of detecting, tracking, and retrieving vehicles based on natural language descriptions.**

This project integrates the power of **YOLOv11** (enhanced with a Coordinate Attention module), **DeepSORT** for robust multi-object tracking, and **OpenAI CLIP** for semantic vehicle retrieval (e.g., searching for *"a light orange and yellow car"*).

---

## Demo

![Demo Preview](assets/demo_tracking.gif)

---

## 📂 Project Structure

The project is organized to separate source code, data, and model artifacts:

```text
Vehicle-Tracking-System/
│
├── datasets/                # Dataset directory (Ignored by Git)
│   ├── merged_dataset_2/    # Processed and merged dataset
│   └── raw/                 # Raw downloaded data
│
├── input/                   # Directory for input videos
│   └── output_tracking.mp4
│
├── output/                  # Directory for processing results
│
├── weights/                 # Model checkpoints
│   ├── best.pt              # The trained YOLOv11+CoordAtt weights
│   └── .gitkeep
│
├── notebooks/               # Source code (Jupyter Notebooks)
│   ├── 01_data_preparation.ipynb    # Data downloading, filtering, and merging
│   ├── 02_training_yolo.ipynb       # Training YOLOv11 with Coordinate Attention
│   └── 03_inference_tracking.ipynb  # Main Inference: Tracking & Retrieval
│
├── assets/                  # Images and GIFs for README
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Key Features

* **Custom Object Detection:** Utilizes a modified **YOLOv11** architecture integrated with **Coordinate Attention (CoordAtt)** mechanisms to enhance feature extraction and spatial awareness for vehicle detection.
* **Robust Multi-Object Tracking:** Implements **DeepSORT** to maintain consistent vehicle IDs even through temporary occlusions.
* **Semantic Retrieval:** Enables searching for specific vehicles using **Natural Language Queries** (Text-to-Image) or Reference Images via the **CLIP** model.
* **Automated Data Pipeline:** A comprehensive pipeline for filtering blurry images, normalizing labels, and merging datasets from multiple sources.

---

## Data & Weights Setup

Due to GitHub's file size limits, the dataset and trained weights are hosted on Google Drive. Please follow the instructions below to set them up.

### 1. Pre-trained Weights (`best.pt`)
This model was trained from scratch for 81 epochs on the custom dataset.

* **Option A: Automatic Download (Recommended)**
    Simply run the `notebooks/inference_tracking.ipynb` notebook. The script includes a block that automatically checks for and downloads the weights if they are missing.

* **Option B: Manual Download**
    1.  Download the file from [Google Drive Link](https://drive.google.com/file/d/1TgD8SUa-hvtHVH-ligQJqrxhzYwzr-Lh/view?usp=sharing).
    2.  Rename the file to `best.pt`.
    3.  Place it inside the `weights/` directory.

### 2. Processed Dataset
The dataset combines two sources (10k + 4k images), filtered for quality and formatted for YOLO.

* **Download Link:** [Google Drive Folder](https://drive.google.com/drive/folders/1LJdaTiqqZZzMvnPrC30hXE-4855bHr4Z?usp=sharing)
* **Instructions:** Download the folder content and extract it into the `datasets/` directory.

---

## Usage Guide

### Step 1: Data Preparation (Optional)
If you wish to reproduce the dataset creation process:
* Open `notebooks/data_preparation.ipynb`.
* The notebook handles downloading raw data, removing blurry images, and merging labels.

### Step 2: Training (Optional)
To retrain the model from scratch:
* Open `notebooks/training_yolo_ca.ipynb`.
* The training pipeline uses **Albumentations** for advanced data augmentation.
* Ensure the `data.yaml` path points to `datasets/merged_dataset_2`.

### Step 3: Inference & Tracking (Main)
To run the system on your own video:

1.  Place your video file (e.g., `traffic_test.mp4`) into the `input/` folder.
2.  Open `notebooks/03_inference_tracking.ipynb`.
3.  Update the configuration in the code:
    ```python
    VIDEO_INPUT = 'input/traffic_test.mp4'
    TEXT_QUERY = 'a white truck' # Change this to search for specific vehicles
    ```
4.  Run all cells. The system will detect, track, and highlight vehicles matching your query.
5.  The output video will be saved in the `output/` folder.

---


## 📊 Results

The model was evaluated on the test set (877 images) after training for 81 epochs. Below are the quantitative results demonstrating the effectiveness of the **YOLOv11 + Coordinate Attention** architecture.

### 1. Detection Metrics (Test Set)

| Class | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes** | 0.631 | 0.751 | **0.633** | 0.404 |
| 🚗 **Car** | **0.857** | **0.749** | **0.819** | **0.490** |
| 🚛 Truck Large | 0.732 | 0.722 | 0.716 | 0.451 |
| 🚚 Truck Medium | 0.746 | 0.683 | 0.716 | 0.464 |
| 🚌 Bus | 0.692 | 0.728 | 0.683 | 0.446 |

> **Highlights:**
> * The model achieves high performance on the **Car** class (**mAP@50: 81.9%**), which is the most frequent object in traffic scenarios.
> * The integration of Coordinate Attention helps in distinguishing large vehicles (Trucks/Buses) with consistent accuracy (> 71% mAP@50).

### 2. Retrieval & Tracking Performance

The system was tested on a video sequence (1870 frames) with a complex natural language query: *"a car light orange and yellow"*.

* **Total Targets Retrieved:** **24 unique vehicle tracks** matched the text description.
* **Average Similarity Score:** ~0.34 (Cosine Similarity with CLIP text embeddings).
* **Tracking Stability:** DeepSORT successfully maintained IDs (e.g., Track ID 175, 179) over long sequences despite occlusions.

### 3. Inference Speed

Benchmarks performed on Google Colab (Tesla T4 & P100):

| Component | Time per Frame | FPS (Approx) |
| :--- | :---: | :---: |
| **YOLOv11 Inference** | 20.8 ms | ~48 FPS |
| **Pre-processing** | 0.4 ms | - |
| **Post-processing** | 1.0 ms | - |

*(Note: Total system FPS including CLIP feature extraction and DeepSORT matching is approximately 25-30 FPS, suitable for real-time applications.)*

---

## Contributing 
We welcome contributions to make this project better! Whether you're fixing a bug, improving documentation, or proposing a new feature, your help is appreciated.

### How to Contribute
1.  **Fork the Repository**: Click the "Fork" button at the top right of this page.
2.  **Clone your Fork**:
    ```bash
    git clone [https://github.com/Group2-CPV301/Vehicle-search-retrieval-by-camera_CPV303.git](https://github.com/Group2-CPV301/Vehicle-search-retrieval-by-camera_CPV303.git)
    ```
3.  **Create a Branch**:
    ```bash
    git checkout -b feature/AmazingFeature
    ```
4.  **Make Changes**: Implement your features or fixes.
5.  **Commit Changes**:
    ```bash
    git commit -m "Add some AmazingFeature"
    ```
6.  **Push to Branch**:
    ```bash
    git push origin feature/AmazingFeature
    ```
7.  **Open a Pull Request**: Go to the original repository and click "New Pull Request".

### Guidelines
* **Code Style**: Please follow standard Python conventions (PEP 8).
* **Large Files**: Do not commit files larger than 100MB (datasets, weights) directly. Use Google Drive links as described in the installation section.
* **Issues**: If you find a bug, please open an issue with details on how to reproduce it.