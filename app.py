# Streamlit app to show recommendations for a user (uses Surprise SVD model)
import streamlit as st
import joblib
from surprise import Dataset, Reader
from surprise import SVD

st.title('CODTECH Recommendation System - Demo')

st.markdown('This demo uses the MovieLens 100k dataset and an SVD model.')

model_path = 'models/svd_model.pkl'

if st.button('Train model (small demo)'):
    st.info('Training a model... this runs the training script in the container environment (if available).')
    st.write('Please run `python src/train.py` locally or on your server to create models/svd_model.pkl.')

user_id = st.text_input('Enter user id (1-943)', value='10')

if st.button('Recommend top 10'):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f'Could not load model at {model_path}. Run `python src/train.py` first. Error: {e}')
    else:
        data = Dataset.load_builtin('ml-100k')
        trainset = data.build_full_trainset()
        all_items = trainset.all_items()
        raw_iids = [trainset.to_raw_iid(i) for i in all_items]
        recs = []
        for iid in raw_iids:
            est = model.predict(user_id, iid).est
            recs.append((iid, est))
        top10 = sorted(recs, key=lambda x: x[1], reverse=True)[:10]
        st.subheader('Top 10 recommendations')
        for rank, (iid, score) in enumerate(top10, start=1):
            st.write(f"{rank}. MovieID {iid} — score {score:.2f}")
