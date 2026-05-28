# 🍌 Banana Quality Classifier: Head-to-Head Architecture

![Banana AI Demo](https://img.shields.io/badge/Status-Completed-success)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

A sophisticated deep learning web application to automatically classify the ripeness level of bananas into three distinct categories: **Mentah (Unripe)**, **Matang (Ripe)**, and **Terlalu Matang (Overripe)**. 

This project demonstrates a direct head-to-head performance comparison between two state-of-the-art Computer Vision architectures: **ConvNeXt-Tiny** and **MobileNetV3-Large**.

## 🚀 Live Demo
*(Insert your Streamlit Cloud link here after deployment)*

## 🔬 Model Performance & Methodology

Both models were trained using **Transfer Learning** on a standardized Kaggle dataset with the implementation of **On-The-Fly Data Augmentation** (Random Flip, Rotation, Zoom) to prevent overfitting and ensure robust generalization against varied backgrounds.

### Validation Results:
- **MobileNetV3-Large**: Achieved **100% Accuracy** on the validation set.
- **ConvNeXt-Tiny**: Achieved **98% Accuracy** on the validation set.

*See the `notebook_klasifikasi_pisang.ipynb` file in this repository for the complete training code, performance graphs, and confusion matrix visualizations.*

## 💻 Tech Stack
- **Framework**: TensorFlow 2.15 (Keras)
- **Frontend UI**: Streamlit 1.32.0 (Custom CSS for Clean White Theme & Tabbed Navigation)
- **Data Pipeline**: `tf.keras.utils.image_dataset_from_directory` with `AUTOTUNE` optimization.

## 🛠️ How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/HyLuthfi/banana-ripeness-classifier.git
   cd banana-ripeness-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Important:** Download your trained `.h5` model weights from Kaggle and place them inside the `models/` directory:
   - `models/best_convnext_pisang.h5`
   - `models/best_mobilenet_pisang.h5`

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
- `app.py`: Main Streamlit application file containing the UI and inference logic.
- `requirements.txt`: Python dependencies.
- `notebook_klasifikasi_pisang.ipynb`: Complete research methodology, training pipeline, and evaluation metrics.
- `models/`: Directory to store the pre-trained `.h5` weight files.

## 🤝 Acknowledgements
Developed by **Luthfi Muthathohirin** for Deep Learning research and portfolio demonstration.
