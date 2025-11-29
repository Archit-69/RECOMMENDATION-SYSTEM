# CODTECH Recommendation System (Task 4)

This repository contains a demonstration notebook and scripts to build a **Recommendation System** using **Collaborative Filtering (Matrix Factorization / SVD)** on the MovieLens 100k dataset.

## Contents
- `notebooks/recommendation_system_task4.ipynb` - Jupyter notebook with step-by-step model training & recommendations.
- `src/train.py` - Training script that fits an SVD model using Surprise and saves it with joblib.
- `app.py` - Simple Streamlit demo to show top-N recommendations for a user (expects `models/svd_model.pkl`).
- `requirements.txt` - Python dependencies.
- `.gitignore` - Suggested ignores for a Python project.

## Quick start (local)
1. Create a virtual environment and install requirements::
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
2. Train model:
   ```bash
   python src/train.py --model_output models/svd_model.pkl
   ```
3. Run Streamlit app (optional):
   ```bash
   streamlit run app.py
   ```

## Notes
- This repo uses the `surprise` library and the built-in `ml-100k` MovieLens dataset.
- The notebook and scripts are ready to be pushed to GitHub as a complete project for the CODTECH internship task.

## License
MIT License — feel free to adapt for your submission.
