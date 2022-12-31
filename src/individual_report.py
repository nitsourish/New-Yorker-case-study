from copyreg import pickle
import os
import tqdm
import pickle
import pandas as pd
import numpy as np
import random
import seaborn as sns
from tqdm import tqdm
import warnings
import argparse
import logging
import datetime
from utility import individual_item_detailed
logger = logging.getLogger()

data_path = '../data/'
report_path = '../data/reports/'

parser = argparse.ArgumentParser(description='reports often or rarely sold together items')
parser.add_argument('-m','--min_cnt', type = int, metavar = '',required=True,default=0,help='minimum order count for eligible article_id')
parser.add_argument('-p','--penetration', type = float, metavar = '',required=True,help='Fraction of unique customers for eligible article_id')
parser.add_argument('-n','--n', type = int, metavar = '',required=True, help='number of often/rarely purchased products for each products')
parser.add_argument('-prod','--product_name', type = str, metavar = '',required=True, help='name of the product for reporting')
args = parser.parse_args()

def data_prep(transaction:pd.DataFrame, products:pd.DataFrame,penetration:float,min_cnt:int=1):

    """
    Data prep function for further Analytics.

    Args:
        params: 
            transaction: Dataframe of the purchases each customer for each date with article id(additional information). 
            products: detailed product hierarchy metadata for each article_id available for purchase
            min_cnt: minimum order count for eligible article_id
            penetration:Fraction of unique customers for eligible article_id
    Returns:
        order instance wise Merged and filtered Dataframe with eligible article/products.
    """
    #For Memory optimization
    transaction['customer_id'] = transaction['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')
    transaction['article_id'] = transaction.article_id.astype('int32')
    products_col = ['article_id','product_code','prod_name','product_type_no','product_type_name', 'product_group_name']
    df = pd.merge(transaction[['t_dat','customer_id','article_id']],products[products_col],on=['article_id'],how='left')
    cnt = pd.DataFrame(df.article_id.value_counts())
    cnt.columns = ['counts']
    if penetration > 0:
        custn = transaction.groupby('article_id')['customer_id'].nunique()
        cust = transaction['customer_id'].nunique()
        custn = custn/cust
        custn = custn.sort_values(ascending=False)
        prod_index_cutoff = list(custn[custn >= penetration].index)
        df = df[df.article_id.isin(prod_index_cutoff)]
    if min_cnt > 1:
        cnt = transaction.article_id.value_counts()
        prod_index_cutoff = list(cnt[cnt >= min_cnt].index)
        df = df[df.article_id.isin(prod_index_cutoff)]         
    df['order_instance'] = df['customer_id'].astype(str) + df['t_dat']
    return df

if __name__ == '__main__':

    logging.basicConfig(filename="output.log",
                        level=logging.DEBUG)
    logger.addHandler(logging.FileHandler('output.log', 'a'))
    print(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"))                     
    train = pd.read_csv(os.path.join(data_path,'transactions_train.csv.zip'))
    products = pd.read_csv(os.path.join(data_path,'articles.csv.zip'))
    print('Data Read')

    df = data_prep(transaction = train,products=products,min_cnt = args.min_cnt,penetration=args.penetration)
    print(f'Data prepared {df.shape}')

    often_rare_purchased = individual_item_detailed(df,products,n=args.n,product_name=args.product_name)
    print(often_rare_purchased)
    # often_rare_purchased.to_csv(os.path.join(report_path,'prod_name_wise_report_individual.csv'),index=False) 