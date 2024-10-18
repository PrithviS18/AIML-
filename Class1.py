import pandas as pd
my_dict = {'Name':['A','B','C','E','A','Z'],
           'Age': [10,12,13,14,10,0],
           'Score': [65,64,63,62,65,0]}
df= pd.DataFrame(my_dict)
print(df)
df['Address'] = ['9A','9B','9C','9D','9E','9F']
print(df)
df.to_csv('my_file.csv')
new_df = pd.read_csv('/content/Q1_data1 (1).txt',header=None,sep=',')
print(new_df)
