#%%
import pandas as pd
import os
'''
data load
'''
df_월세사례정보 = pd.read_csv('../data/월세임대 사례정보.csv', encoding='cp949')
df_월세예측시세 = pd.read_csv('../data/월세임대 예측시세.csv', encoding='cp949')
df_전세사례정보= pd.read_csv('../data/전세임대 사례정보.csv', encoding='cp949')
df_전세예측시세 = pd.read_csv('../data/전세임대 예측시세.csv', encoding='cp949')
df_주택기본정보 = pd.read_csv('../data/주택기본정보.csv', encoding='cp949')
df_주택배치도정보 = pd.read_csv('../data/주택배치도정보.csv', encoding='cp949')

#%%
data_dir = '../data/'
data_list = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.csv')]
print(data_list)
df_list = [df_월세사례정보, df_월세예측시세, df_전세사례정보, df_전세예측시세,df_주택기본정보, df_주택배치도정보]
#%%
'''
0. 길이 확인
'''
for i in range(len(df_list)):
    print(data_list[i], ' : ', len(df_list[i]), end='\n')

#%%
df_주택기본정보['경도(LNG)']
df_주택기본정보['위도(LAT)']
df_주택기본정보.columns
#%%
df_월세예측시세.columns