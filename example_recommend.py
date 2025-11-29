# Example: load model and print top-10 recommendations for a given user
import joblib
from surprise import Dataset

model = joblib.load('models/svd_model.pkl')
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
all_items = trainset.all_items()
raw_iids = [trainset.to_raw_iid(i) for i in all_items]

user_id = '10'
recs = [(iid, model.predict(user_id, iid).est) for iid in raw_iids]
top10 = sorted(recs, key=lambda x: x[1], reverse=True)[:10]
for r in top10:
    print(r)
