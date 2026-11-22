# Credit card fraud detection with PyOD

This experiment uses the **exact version 1 CSV** from [the specified Kaggle dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection). It is not a substitute dataset. The script checks the CSV's SHA-256, column names, row count and fraud count before training. The CSV is excluded from Git because it is approximately 151 MB and is distributed by Kaggle.

With Python 3.11 installed, from this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
curl -fL 'https://www.kaggle.com/api/v1/datasets/download/whenamancodes/fraud-detection' -o data/fraud-detection.zip
unzip -o data/fraud-detection.zip creditcard.csv -d data
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python fraud_autoencoder.py
```

For editor diagnostics, open the repository folder as the project and select the Python interpreter in its virtual environment (`.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows). In VS Code, use **Python: Select Interpreter**; in PyCharm or IntelliJ with the Python plugin, set the project Python interpreter under **Settings > Project > Python Interpreter**. If imports are underlined in the editor, confirm it is using this interpreter rather than the system Python.

If Kaggle requires a login or the API download is unavailable, download version 1 from the linked Kaggle page and extract `creditcard.csv` into `data/`. The program rejects a different file rather than reporting results for the wrong dataset. On Windows, activate the virtual environment with `.venv\Scripts\activate` and download/extract the dataset in a browser; run `python fraud_autoencoder.py` from the project directory.

Training uses the 30 numeric transaction features; `Class` is held out from model fitting and used only to stratify the split and assess test results. A training-only RobustScaler normalizes the inputs. The unsupervised AutoEncoder assigns higher scores to transactions it reconstructs less accurately; the training-score threshold uses a preset 0.002 contamination fraction. Average precision is reported alongside the random-precision baseline because ordinary accuracy hides failures on this imbalanced dataset. Results can vary by platform and package version even with a fixed seed.
