import requests
import pandas as pd
import json
import re
from datetime import datetime, timedelta
from tqdm import tqdm
import os

def related_queries(quries):
    if quries == []:
        res = ''
    else:
        res = ''
        for related_query in quries:
            res += res + related_query['query'] + '; '
    return res

def google_trends_crawl(date: str, country: str):
    url = f'https://trends.google.com.tw/trends/api/dailytrends?hl=en&tz=-480&geo={country}&ns=15&ed={date}'
    resp = requests.get(url)
    try:
        df = pd.DataFrame(json.loads(re.sub(r'\)\]\}\',\n', '', resp.text))['default']['trendingSearchesDays'][0]['trendingSearches'])

        df = df.drop(columns=['shareUrl', 'image', 'articles'])

        df['title'] = df['title'].apply(lambda x: x['query'])
        df['relatedQueries'] = df['relatedQueries'].apply(related_queries)
        df.columns = ['topic', 'traffic', 'sub_topics']
        return df
    except:
        return None


end_date = datetime.today()
start_date = end_date - timedelta(days=300)
str_end_date = datetime.strftime(end_date, '%Y%m%d')
str_start_date = datetime.strftime(start_date, '%Y%m%d')


output_file_path = f'./output/google_trends/google_trends_{str_start_date}_to_{str_end_date}.csv'
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)


for i in tqdm(pd.date_range(start=start_date, end=end_date, freq='1D')):
    str_i_date = datetime.strftime(i, '%Y%m%d')
    ndf = google_trends_crawl(str_i_date, 'US')
    if ndf is None:
        print(f'{str_i_date} is None')
        continue
    ndf['date'] = str_i_date
    if not os.path.exists(output_file_path):
        ndf.to_csv(output_file_path, index=False)
    else:
        ndf.to_csv(output_file_path, mode='a', header=False, index=False)

df_us = pd.read_csv(output_file_path)
pass