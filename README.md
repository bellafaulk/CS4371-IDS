# AI-powered Intrusion Detection System (IDS)

## Overview
This project simulates an home Intrusion Detection System (IDS) with AI-powered security, utilizing a neural network to detect any anomalies in a dataset. Our project was inspired by the research completed on how AI can strengthen residential or home security systems.

## Advanced Features
* **Contextual Anomaly Detection:** The model is trained on feature-augmented data, including a historical intrusion density feature (P_avg). This is so the system can detect threats based on historical patterns over the last N time steps, not on static input.
* **Model Robustness:** We implemented L2 regularization and dropout in the deep learning architecture to prevent overfitting and make sure the model generalizes successfully.
* **Security Evaluation:** Our system is evaluated by security industry standard (Precision, Recall, and F1-score) and uses a fine-tuned detection threshold (0.30).
* **Imbalanced Data Handling:** We implemented dynamic class weighting to train the model on rare intrusion events (only five percent of data), so the model can effectively include the threat class.
* **Visualization and Logging:** Generates a confusion matrix for visual understanding and includes real-time logging of all alerts to 'intrusion_log.txt'.


## Team Members
- Angelica Barrientos
- Jackie Medrano
- Bella Faulk
- Megan Botha
- John Parsons

## Project Setup and Environment

### 1. Install Python 3.12

  Since TensorFlow only supports up to Python 3.12, download and install this version of Python 3.12.10 from https://www.python.org/downloads/

### 2. Clone the repository

  Clone the repository and navigate to the project directory.
  ```bash
  git clone https://github.com/bellafaulk/CS4371-IDS.git
  cd CS4371-IDS
  ```

### 3. Create a virtual environment

  Create an isolated environment for certain dependencies and activate.
  ```bash
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 4. Install dependencies
  Install all required libraries, including visualization and deep learning packages.
  ```bash
  python -m pip install -r requirements.txt
  ```

### 5. Run the IDS neural network
  Run the project's main prototype. The script trains the complex model, evaluates performance, and runs a real-time simulation.
  ```bash
  python ids_model.py
  ```
  The system will output the Precision/Recall results and generate the IDS Condusion Matrix figure, verifying its effectiveness.

