import argparse

parser = argparse.ArgumentParser(description='enter lg')
parser.add_argument('--lg', help='language')
args = parser.parse_args()
lg = args.lg
chunksize = 4

from tqdm.auto import tqdm
from easynmt import EasyNMT
import pandas as pd
import torch

def BackTranslation(df, lg):
    sentences_en = df.text.to_list()
    translations_lg = model.translate(sentences_en, target_lang=lg)
    translations_back_translated = model.translate(translations_lg, target_lang='en')
    return translations_back_translated


df = pd.read_csv('parler-hate-speech/parler_annotated_data.csv')
df_in_chunks = pd.read_csv('parler-hate-speech/parler_annotated_data.csv', chunksize=chunksize)

# for lg in ['de','fr','ko', 'he', 'es']:
print(lg)
dest_csv = f'parler-hate-speech/parler_annotated_data_{lg}.csv'
assert 1 == 0

model = EasyNMT('m2m_100_418M')
df_backtraslated = pd.DataFrame()
for df_chunk in tqdm(df_in_chunks, total= len(df)//chunksize):    
    try:
        df_chunk[f'BackTranslation_{lg}'] = BackTranslation(df_chunk,lg)
        df_backtraslated = pd.concat([df_backtraslated, df_chunk])
        df_backtraslated.to_csv(dest_csv, index=False)
    except Exception as e:
        print(e)
            