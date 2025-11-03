# AI-powered Intrusion Detection System (IDS)

## Overview
This project simulates an home Intrusion Detection System (IDS) with AI-powered security, utilizing a neural network to detect any anomalies in sensor data. Our project was inspired by the research completed on how AI can strengthen residential or home security systems.

## Team Members
- Angelica Barrientos
- Jackie Medrano
- Bella Faulk
- Megan Botha
- John Parsons

## Project Setup and Environment

1. Install Python 3.12

Since TensorFlow only supports up to Python 3.12, download and install this version of Python 3.12.10 from https://www.python.org/downloads/

2. Clone the repository

Clone the repository from GitHub.
```bash
git clone https://github.com/bellafaulk/CS4371-IDS.git
cd CS4371-IDS
```

3. Create a virtual environment

Create an isolated environment for certain dependencies.
```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies
Install all required libraries and make sure pip is upgraded.
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

5. Verify installation
Check that everything has been installed correctly.
```bash
python3 -m pip list
```
You should see:
```bash
tensorflow
numpy
pandas
scikit-learn
matplotlib
```
Test TensorFlow:
```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```
If this command prints the version and doesn't show any errors, then your environment is ready!

6. Run the IDS neural network
Run the project's main prototype.
```bash
python3 src/ids_model.py
```

