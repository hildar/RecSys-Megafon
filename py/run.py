# answers_test
import pandas as pd
import dill
# import lightgbm
# import pyarrow
# import fastparquet

# load test file with features
test_feats = pd.read_parquet('test.parquet')

# load model
with open('lgb_gsc.dill', 'rb') as f:
    lgb_gsc = dill.load(f)

# predict
train_preds = lgb_gsc.predict_proba(test_feats)

# prepare new df
answers = test_feats.iloc[:, :3]

# write targets
answers['target'] = train_preds[:, -1]

# save df to csv
answers.to_csv('answers_test.csv', index=False)
