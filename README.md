# Real-time Driver Drowsiness Detection (CNN + Facial Landmarks)

A driver drowsiness detector that watches a webcam feed and sounds a beep alarm when the driver's **eyes stay closed** or when they **yawn**.

> **Credit:** this is a fork of
> [Pratham-mehta/Real-time-Driver-Drowsiness-Detection-System-Using-Deep-Learning](https://github.com/Pratham-mehta/Real-time-Driver-Drowsiness-Detection-System-Using-Deep-Learning),
> the CS-GY 6953 Deep Learning major project. The training notebook, architecture diagrams, project report and beep sound come from the original repo. See [What I added](#what-i-added) for my changes.

## How it works

| Signal | Method |
|---|---|
| Eyes open / closed | CNN (trained in the notebook) classifies a crop of each eye, located with 68-point facial landmarks |
| Yawning | Mouth aspect ratio (MAR) computed from the inner-mouth landmarks |
| Alarm | Beep plays while eyes stay closed for 1.5 s, or the mouth stays wide open for 1.2 s |

Pipeline: webcam frame → Haar face detector → OpenCV Facemark (LBF) landmarks → eye crops (CNN) + MAR → alarm logic → beep.

## Dataset

[Drowsiness dataset on Kaggle](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) with four classes: `Closed`, `Open` (eye images) and `yawn`, `no_yawn` (face images). The original README attributes the images to the YawDD and CEW datasets. After processing, the notebook builds a set of 1,925 labelled images, resized to 145×145.

## Results

From the training notebook (50 epochs):

| Metric | Value |
|---|---|
| Max test accuracy | 97.58% |
| Max training accuracy | 96.73% |
| Min test loss | 0.0712 |
| Min training loss | 0.0888 |

In informal tests on a laptop webcam, the eye-closure and yawn alarms triggered as expected.

### Limitations

* The dataset is small and the train/test split is random, so near-duplicate frames can appear in both sets. Real-world accuracy is likely lower than 97.58%.
* The CNN's yawn output did **not** transfer to a laptop webcam (it gave the same answer with the mouth open or closed), so yawning is detected with landmarks instead. The CNN is used for eyes only.
* Face detection uses a Haar cascade, so it works best with a mostly frontal face and decent lighting. Glasses, strong head turns and dim light can reduce accuracy.
* The thresholds (`CLOSED_THRESH`, `YAWN_MAR_THRESH`, hold times) were set by hand and may need tuning for other cameras.

## Run it

Tested with **Python 3.12** on Windows. TensorFlow does not yet publish wheels for Python 3.14, so use 3.12 (3.10 to 3.13 should also work).

```bash
# 1. create and activate an environment
py -3.12 -m venv drowsy_env
drowsy_env\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. download the facial-landmark model (about 54 MB, not stored in this repo)
curl -L -o lbfmodel.yaml https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml

# 4. run
python drowsiness_webcam.py
```

Press `q` to quit. Make sure `drowsiness_model.keras`, `beep-warning-6387.mp3` and `lbfmodel.yaml` are in the same folder as the script.

## Retraining

Open `Final_Project_Driver_Drowsiness_Detection.ipynb` in Google Colab (T4 GPU), run it top to bottom, then download the saved `drowsiness_model.keras`. The notebook downloads the Kaggle datasets with the Kaggle CLI, so you need a Kaggle account and API credentials.

## What I added

* **`drowsiness_webcam.py`**: the real-time webcam program, with beep alarm and on-screen status and measurements.
* **Landmark-based yawn detection** using the mouth aspect ratio, replacing the CNN's yawn output after it failed on a live webcam.
* **Landmark-based eye crops** for the CNN, so eyes are still found when closed.
* **Notebook fixes for current Colab/Python:**
  * downloads use the Kaggle CLI, since `opendatasets` breaks on Python 3.13 (the `cgi` module was removed);
  * OpenCV pinned below version 5, which dropped `CascadeClassifier`;
  * `np.array(..., dtype=object)` for the mixed image/label list;
  * `ImageDataGenerator` imported from Keras's legacy path;
  * the model saved in the `.keras` format.
* **`requirements.txt`** and this README.

## Files

| File | Purpose |
|---|---|
| `Final_Project_Driver_Drowsiness_Detection.ipynb` | Training notebook |
| `drowsiness_model.keras` | Trained CNN |
| `drowsiness_webcam.py` | Real-time detector |
| `beep-warning-6387.mp3` | Alarm sound |
| `Architecture Diagram_*.png`, `FinalProjectReport_DL_DriverDrowsiness.pdf` | From the original project |
| `requirements.txt` | Python dependencies |

## License

See `LICENSE` (inherited from the original repository).
