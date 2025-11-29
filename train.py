"""Train a recommendation model using Surprise (SVD) on MovieLens 100k.
Usage:
    python src/train.py --model_output models/svd_model.pkl
"""
import os
import argparse
import joblib
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import surprise

def train_and_save(output_path):
    data = Dataset.load_builtin('ml-100k')
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=50, random_state=42)
    print('Training model...')
    model.fit(trainset)
    print('Evaluating...')
    predictions = model.test(testset)
    surprise.accuracy.rmse(predictions, verbose=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f'Model saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_output', default='models/svd_model.pkl', help='Path to save trained model')
    args = parser.parse_args()
    train_and_save(args.model_output)
