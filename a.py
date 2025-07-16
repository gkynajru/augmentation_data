import pandas as pd

df = pd.read_csv('data/output/review/reviewed_samples_20250717_221849.csv')
df.drop(columns='Unnamed: 0', inplace=True, axis=1)
df.to_csv('data/output/review/reviewed_samples_20250717_221849.csv', index=False)