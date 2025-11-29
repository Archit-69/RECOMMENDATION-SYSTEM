import traceback

try:
    import surprise
    from surprise import SVD, Dataset
    from surprise.model_selection import train_test_split as sp_split

    def main():
        data = Dataset.load_builtin('ml-100k')
        trainset, testset = sp_split(data, test_size=0.2)

        model = SVD()
        model.fit(trainset)

        user_id = '10'
        all_items = trainset.all_items()
        item_ids = [trainset.to_raw_iid(i) for i in all_items]

        # collect items the user has already rated (item/raw id is at index 1 in testset tuples)
        user_rated = [j[1] for j in testset if j[0] == user_id]

        recs = []
        for iid in item_ids:
            if iid in user_rated:
                continue
            est = model.predict(user_id, iid).est
            recs.append((iid, est))

        top10 = sorted(recs, key=lambda x: x[1], reverse=True)[:10]
        # Try to map raw item ids to MovieLens titles (u.item)
        import os
        try:
            import pandas as pd
        except Exception:
            pd = None

        ml_dir = os.path.join(os.path.expanduser('~'), '.surprise_data', 'ml-100k')
        item_file = os.path.join(ml_dir, 'u.item')
        id2title = None
        if pd is not None and os.path.exists(item_file):
            try:
                items = pd.read_csv(item_file, sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movie_id','title'])
                id2title = dict(zip(items['movie_id'].astype(str), items['title']))
            except Exception:
                id2title = None

        print("Top 10 recommendations (title or id, score):")
        for iid, score in top10:
            title = id2title.get(iid) if id2title else None
            label = title if title is not None else iid
            print(label, score)

    if __name__ == '__main__':
        main()

except Exception:
    traceback.print_exc()
