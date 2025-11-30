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
* **Paper Title:** [Enhancing Residential Security with AI-Powered Intrusion Detection Systems](https://drive.google.com/file/d/1uJjXHR5O8E4biDnBv9fDr-hHcwK_lrRe/view)
* **Key Contributions:** The research vouches for replacing traditional home security systems with an AI-powered IDS (Intrusion Detection System) that uses multiple data inputs (cameras, motion, doors). The main technical contribution is identifying the Convolutional Neural Network (CNN) as the best model for accurate real-time anomaly detection, which achieved 97.34% accuracy and high performance (94.24% F1-Score) on sensor patterns.

### 2. Contemporary Work - Building upon Findings
* **Paper Title:** [Real-Time Intrusion Detection in Smart Home Environments Through Federated Deep Learning on IoT Edge Devices (2025)](https://www.researchgate.net/publication/396155650_Real-Time_Intrusion_Detection_in_Smart_Home_Environments_Through_Federated_Deep_Learning_on_IoT_Edge_Devices)
* **Connection:** While our project establishes a central AI model for intrusion detection, this contemporary research introduces **federated learning**. It addresses the critical "future work" challenge of **data privacy** by training the AI locally on edge devices (like smart home hubs) instead of transmitting sensitive sensor data to a central server, building upon our goal of secure, private residential monitoring.

## Functionality and System Status

### Implemented Features (Working)
* **AI-Powered IDS Core:** A deep Neural Network utilizing TensorFlow and Scikit-learn to classify sensor data as 'Normal' or 'Intrusion'.
* **Contextual Anomaly Detection:** The model uses a calculated historical intrusion density feature ($\text{P}_{\text{avg}}$) to provide sequence-aware detection.
* **Imbalance Handling:** Dynamic class weighting is applied during training to effectively learn from the rare $5\%$ intrusion events.
* **Real-time Alerting & Logging:** The system provides a real-time simulation, printing alerts to the console and logging all intrusion events to `intrusion_log.txt`.
* **Robust Evaluation:** Performance is measured using security standard regulation (Precision, Recall, and F1-Score), and visualized with a Confusion Matrix figure saved as `ids_confusion_matrix.png`.

### Non-Implemented Features (Future Work)
* **Live Sensor Integration:** The system currently relies on synthetic data and a simulation buffer; it does not yet connect to live home security sensors (e.g., motion detectors, camera feeds).

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

