import pandas as pd
import numpy as np

df = pd.read_csv('/content/Q2_bollywood.csv')
df.head()
pd.crosstab(df['Genre'],df['ReleaseTime'])
df.head(20)
for i in df['Release Date']:
  print(i.split('-')[1])
df.groupby(['Month','Genre'])['BoxOfficeCollection'].mean()
df[df['BoxOfficeCollection']>100]
df.drop('SlNo',axis=1,inplace=True)
import seaborn as sns
sns.boxplot(df['YoutubeViews'])
Q1 = df['YoutubeViews'].quantile(0.25)
Q3 = df['YoutubeViews'].quantile(0.75)
Q2 = df['YoutubeViews'].median()
IQR = Q3-Q1
upper_whisker = Q3 + 1.5*IQR
lower_whisker = Q1 - 1.5*IQR

outlier = df[(df['YoutubeViews']>upper_whisker)|(df['YoutubeViews']<lower_whisker)]

print(outlier)
