# English and Hindi Alphabet Recognition Project

This project focuses on building a computer vision model to recognize handwritten English and Hindi alphabets using deep learning. The project includes data processing scripts, analysis of the dataset, and the implementation deep learning models for alphabet recognition for both English and Hindi Langauages.

## Project Setup

### Dataset
The dataset used in this project includes:
- **English Dataset**: Handwritten alphabet images obtained from [A-Z Handwritten Alphabets in .csv format Kaggle](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/data).
- **Hindi Dataset**: Handwritten Devanagari characters obtained from [Devanagari Handwritten Character Dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset).

### Data Processing
To prepare the data for model training, the following steps were performed:
1. **English Dataset**: A script (`csv_to_images.py`) was used to convert the CSV data containing pixel values for English alphabets into images. The script reshapes the data, encodes labels, and saves images to folders with subfolders for each alphabet class.
2. **Hindi Dataset**: The Devanagari characters dataset was structured with separate subfolders for each character in both training and testing sets.

### Combining Datasets
Both English and Hindi datasets were combined into a unified dataset with folders for training and testing images. The combined dataset is structured to facilitate model training on alphabets from both languages.

## Usage
1. Download the datasets from the provided sources.
2. Run the `csv_to_images.py` script for the English dataset to convert CSV data to images.
3. Combine the English and Hindi datasets into a unified dataset structure.

## Project Structure
- **`csv_to_images.py`**: Script to convert CSV data to images for the English alphabet dataset.
- **`english_alphabets_data/`**: Folder containing the combined English and Hindi alphabet datasets.
  - **`train/`**: Training images organized by class (alphabet).
  - **`test/`**: Testing images organized by class (alphabet).


## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- TensorFlow(for the alphabet recognition model)


