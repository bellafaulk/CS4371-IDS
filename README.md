# AI-powered Intrusion Detection System (IDS)

## Overview
This project simulates an home Intrusion Detection System (IDS) with AI-powered security, utilizing a neural network to detect any anomalies in a dataset. Our project was inspired by the research completed on how AI can strengthen residential or home security systems.

## Team Members
- Angelica Barrientos
- Jackie Medrano
- Bella Faulk
- Megan Botha
- John Parsons

## Academic Lineage and Research Context

This project aligns our work with the academic research and journey of Intrusion Detection Systems (IDSs).

### 1. Prior Research - Foundational Bedrock
This paper served as the direct inspiration and foundational blueprint for our approach, validating the shift from traditional security to AI-based anomaly detection:

* **Paper Title:** Enhancing Residential Security with AI-Powered Intrusion Detection Systems
* **Key Contributions:** The research vouches for replacing traditional home security systems with an AI-powered IDS (Intrusion Detection System) that uses multiple data inputs (cameras, motion, doors). The main technical contribution is identifying the Convolutional Neural Network (CNN) as the best model for accurate real-time anomaly detection, which achieved 97.34% accuracy and high performance (94.24% F1-Score) on sensor patterns.

### 2. Contemporary Work

* **Paper Title:** 

## Functionality and System Status

### Implemented Features (Working)
* **AI-Powered IDS Core:** A deep Neural Network utilizing **TensorFlow** and **Scikit-learn** to classify sensor data as 'Normal' or 'Intrusion'.
* **Contextual Anomaly Detection:** The model uses a calculated historical intrusion density feature ($\text{P}_{\text{avg}}$) to provide sequence-aware detection.
* **Imbalance Handling:** Dynamic class weighting is applied during training to effectively learn from the rare $5\%$ intrusion events.
* **Real-time Alerting & Logging:** The system provides a real-time simulation, printing alerts to the console and logging all intrusion events to `intrusion_log.txt`.
* **Robust Evaluation:** Performance is measured using security-standard **Precision, Recall, and F1-Score**, and visualized via a **Confusion Matrix** figure saved as `ids_confusion_matrix.png`.

### Non-Implemented Features (Future Work)
* **Live Sensor Integration:** The system currently relies on synthetic data and a simulation buffer; it does not yet connect to live home security sensors (e.g., motion detectors, camera feeds).
* **REST API:** No dedicated API endpoint exists for external applications to submit data for inference (currently executed via a single Python script).

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

