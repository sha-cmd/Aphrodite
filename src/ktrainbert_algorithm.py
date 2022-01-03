import ktrain
from ktrain import text
import pandas as pd

X_train_name = 'data/X_train.csv'
y_train_name = 'data/y_train.csv'
X_test_name = 'data/X_test.csv'
y_test_name = 'data/y_test.csv'

df = pd.DataFrame([pd.read_csv(X_train_name, index_col='index')['tweets'],
                   pd.read_csv(y_train_name, index_col='index')['note']]).T
df = df[:70000]
df['note'] = df['note'].map({0: 'negative', 4: 'positive'})
df_val = pd.DataFrame([pd.read_csv(X_test_name, index_col='index')['tweets'],
                       pd.read_csv(y_test_name, index_col='index')['note']]).T
df_val['note'] = df_val['note'].map({0: 'negative', 4: 'positive'})
df_val = df[:10000]
trn, val, preproc = text.texts_from_df(train_df=df, val_df=df_val,
                                       text_column='tweets',
                                       label_columns=['note'],
                                       maxlen=100,
                                       max_features=100000,
                                       preprocess_mode='bert',
                                       val_pct=0.1)

model = text.text_classifier(name='bert', train_data=trn, preproc=preproc, metrics=['accuracy'])
learner = ktrain.get_learner(model=model,
                             train_data=trn,
                             val_data=val,
                             batch_size=32,
                             use_multiprocessing=True,
                             workers=4)

learner.fit_onecycle(lr=2e-5, epochs=1, checkpoint_folder='output')
