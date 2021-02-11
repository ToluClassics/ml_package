import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
MODEL = os.environ.get("MODEL")
FOLD = int(os.environ.get("FOLD"))

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

#this would execute when the code is run
if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    #assign target variable
    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    #assign feature variable
    train_df = train_df.drop(['id','target','kfold'], axis=1)
    valid_df = valid_df.drop(['id','target','kfold'], axis=1)

    valid_df = valid_df[train_df.columns]

    #encode columns
    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c,lbl))
    
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df,ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    print(metrics.roc_auc_score(yvalid,preds))

    joblib.dump(label_encoders, f"models/{MODEL}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}.pkl")
    print(f"done serializing model {MODEL}")