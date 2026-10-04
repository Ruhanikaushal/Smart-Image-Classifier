# 🚀 Complete How-To-Run Guide: Smart Image Classifier

This guide provides end-to-end instructions for running the project on your local computer (Windows, macOS, or Linux) and deploying it live on **Render**.

---

## 📋 Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Local Setup (Step-by-Step)](#2-local-setup-step-by-step)
3. [Running the Interactive Web App (Streamlit)](#3-running-the-interactive-web-app-streamlit)
4. [Running CLI Inference on Test Images](#4-running-cli-inference-on-test-images)
5. [Re-generating Metrics & Confusion Matrix Plots](#5-re-generating-metrics--confusion-matrix-plots)
6. [Deploying Live on Render (Step-by-Step)](#6-deploying-live-on-render-step-by-step)
7. [Troubleshooting & FAQs](#7-troubleshooting--faqs)

---

## 1. Prerequisites

Before starting, ensure you have:
- **Python 3.10 or 3.11** installed. Check via:
  ```bash
  python --version
  ```
- **Git** installed on your system. Check via:
  ```bash
  git --version
  ```
- **Pip** package manager updated:
  ```bash
  python -m pip install --upgrade pip
  ```

---

## 2. Local Setup (Step-by-Step)

### Step 2.1: Open your Terminal / PowerShell
Navigate to your project directory:
```bash
cd "E:\Smart Image Classifier"
```
*(Or the location where your folder resides).*

### Step 2.2: (Recommended) Create and Activate a Virtual Environment
Using a virtual environment isolates project dependencies:

- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays a script execution policy error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again).*

- **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 2.3: Install Required Dependencies
Install all required libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```

Verify that the core dependencies installed properly:
```bash
python -c "import torch, torchvision, streamlit; print('All core libraries loaded successfully!')"
```

---

## 3. Running the Interactive Web App (Streamlit)

To launch the web interface locally:

```bash
streamlit run app.py
```

### What Happens:
1. Streamlit starts a local development server.
2. In your terminal, you will see output like:
   ```
     Local URL: http://localhost:8501
     Network URL: http://192.168.x.x:8501
   ```
3. Your default web browser will open automatically.
4. **Features inside the UI:**
   - **🌱 Live Diagnosis:** Select any preloaded sample test image OR drag-and-drop your own plant leaf image.
   - **📊 Top-5 Predictions:** Real-time confidence scores and probability bar charts.
   - **💊 Agronomic Advice:** Pathogen classification and treatment steps.
   - **📈 Metrics Tab:** View the training loss/accuracy curves and confusion matrix heatmaps.
   - **🔬 Error Analysis Tab:** Inspect the documented failure case (blurred/underexposed tomato leaf misclassification).

To stop the web app, press `Ctrl + C` in your terminal.

---

## 4. Running CLI Inference on Test Images

If you prefer testing the model directly through the command line:

```bash
python Documents/test_inference.py
```

### What This Does:
1. Loads `best_plant_model.pth` and `classes.json`.
2. Evaluates all images inside the `test_images/` directory.
3. Prints the top-3 predicted candidate classes with exact percentage confidence.
4. Generates and saves a visual grid plot named `Documents/test_predictions_grid.png`.

---

## 5. Re-generating Metrics & Confusion Matrix Plots

To re-generate the figures used in the documentation:

### 1. Training & Validation Curves:
```bash
python plot_training_curves.py
```
*Output: Generates `training_curves.png` showing Cross-Entropy Loss and Accuracy over 23 epochs with the Early Stopping point.*

### 2. Confusion Matrix Heatmaps & Classification Report:
```bash
python plot_confusion_matrix.py
```
*Output: Generates `confusion_matrix_full.png`, `confusion_matrix_focused.png`, and updates `metrics_report.md`.*

---

## 6. Deploying Live on Render (Step-by-Step)

[Render](https://render.com) is a modern cloud hosting platform with a free tier. We have already prepared `render.yaml` and `.env.example` to make deployment seamless.

### Step 6.1: Push Your Code to GitHub
1. Initialize git and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Smart Image Classifier with Streamlit and PyTorch"
   ```
2. Create a new public repository on [GitHub](https://github.com/new) (e.g., `smart-image-classifier`).
3. Push your repository:
   ```bash
   git remote add origin https://github.com/<your-username>/smart-image-classifier.git
   git branch -M main
   git push -u origin main
   ```
   *(Note: The model weight file `best_plant_model.pth` is ~18.3 MB, well below GitHub's 100 MB limit, so it pushes smoothly without needing Git LFS).*

### Step 6.2: Deploy on Render
1. Go to [Render Dashboard](https://dashboard.render.com/) and sign in with GitHub.
2. Click **New +** ➔ Select **Web Service**.
3. Select your repository: `smart-image-classifier`.
4. Configure the settings:
   - **Name:** `smart-image-classifier`
   - **Region:** Choose the region closest to you (e.g., Singapore or Oregon).
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command:**
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
     ```
   - **Instance Type:** `Free` (0.5 CPU, 512 MB RAM)

### Step 6.3: Add Environment Variables
Under the **Environment Variables** section on Render, add:
- `PYTHON_VERSION` = `3.11.9`
- `STREAMLIT_SERVER_HEADLESS` = `true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS` = `false`
- `MODEL_PATH` = `best_plant_model.pth`
- `CLASSES_PATH` = `classes.json`

*(You can refer to `.env.example` for reference).*

### Step 6.4: Click "Create Web Service"
- Render will pull your repository, install dependencies, and launch your Streamlit web service.
- Within 2-3 minutes, you will receive a live URL:  
  `https://smart-image-classifier.onrender.com`

---

## 7. Troubleshooting & FAQs

### Q1: `ModuleNotFoundError: No module named 'torchvision'`
**Fix:** Run:
```bash
pip install torchvision --upgrade
```

### Q2: Port 8501 is already in use
**Fix:** Specify an alternative port:
```bash
streamlit run app.py --server.port 8502
```

### Q3: `cv2` or `libGL.so.1` error on Linux / Render
**Fix:** Use `opencv-python-headless` instead of `opencv-python`. This is already configured in our `requirements.txt`.

### Q4: Out-of-Memory (OOM) on free cloud tiers
**Fix:**
PyTorch can allocate extra memory buffers. In `classifier.py`, all inferences are wrapped with:
```python
with torch.no_grad():
    ...
```
This disables gradient computation and keeps memory consumption under 250 MB.
