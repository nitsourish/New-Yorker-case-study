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

### Data prep for Forecasting ###

def data_prep(transaction:pd.DataFrame, products:pd.DataFrame, cum_sales_fraction:float=0.0,penetration:float=0.0):

    """
    Over all Data prep function for forecasting.

    Args:
        params: 
            transaction: Dataframe of the purchases each customer for each date with article id(additional information). 
            products: detailed product hierarchy metadata for each article_id available for purchase
            cum_sales_fraction: fraction of cumulative sales for filtering top products  
            penetration:Fraction of unique customers for eligible article_id
    Returns:
        order instance wise Merged and filtered Dataframe with eligible article/products.
    """

    # transaction['customer_id'] = transaction['customer_id'].apply(lambda x: int(x[-16:],16) ).astype('int64')
    # transaction['article_id'] = transaction.article_id.astype('int32')
    products_col = ['article_id','product_code','prod_name']
    df = pd.merge(transaction[['t_dat','customer_id','article_id','price']],products[products_col],on=['article_id'],how='left')
    if penetration > 0:
        custn = df.groupby('product_code')['customer_id'].nunique()
        cust = df['customer_id'].nunique()
        custn = custn/cust
        custn = custn.sort_values(ascending=False)
        prod_index_cutoff = list(custn[custn >= penetration].index)
        df = df[df.product_code.isin(prod_index_cutoff)]
        print(df.shape)
    if cum_sales_fraction > 0:
        print('entered')
        prod_sales = df.groupby('product_code').price.sum()
        prod_sales=prod_sales.sort_values(ascending=False)
        total_sales = sum(prod_sales)
        prod_sales = prod_sales.cumsum()
        prod_sales = prod_sales/total_sales
        prod_index_cutoff = list(prod_sales[prod_sales <= cum_sales_fraction].index)
        df = df[df.product_code.isin(prod_index_cutoff)]         
    # df['order_instance'] = df['customer_id'].astype(str) + df['t_dat']
    print(f'unique products: {len(prod_index_cutoff)}')
    return df

### individual_item level Report ###    

def individual_item_detailed(df:pd.DataFrame,products:pd.DataFrame,product_name:str,n:int):

    """
    Reporting function to generate detailed analytics of frequently and rarely product combination along with metrics for an individual item.
    Although can be generated from all product report,added flexibility to create for any numbers of items

    Args:
        params: 
            df: order instance wise Merged and filtered Dataframe with eligible article/products 
            product_name: product name
            n: number of often/rarely purchased products to return for each products
            product_df:detailed product hierarchy metadata for each article_id available for purchase
            support:dictionary consisting individual items support score

    Returns:
        a) Detailed report of list(based on n) of frequently/rarely co-purchased items for each item
        along with support for respective item and confidence metrics for each frequently co-purchased items.
    
    """
    n_order = df['order_instance'].nunique()
    orders = df.loc[df['prod_name']==product_name,'order_instance'].unique()
    order_cnt = df.groupby(['prod_name'])['order_instance'].nunique()
    support = order_cnt/n_order
    support=support.to_dict()
    most_frequent = {}
    most_frequent_cnt = {}
    item_order_cnt = {}
    least_frequent = {}
    cnt2 = df.loc[(df.order_instance.isin(orders))&(df['prod_name']!=product_name),'prod_name'].value_counts()
    most_frequent[product_name] = [cnt2.index[j] for j in range(1,n+1)]
    least_frequent[product_name] = [cnt2.index[j] for j in range(-1,-(n+1),-1)]
    most_frequent_cnt[product_name] = [cnt2.values[j] for j in range(1,n+1)]
    item_order_cnt[product_name] = len(orders)
    often_rare_purchased = pd.DataFrame({'prod_name':list(most_frequent.keys()),'item_order_cnt':item_order_cnt.values(),'often_purchased':list(most_frequent.values()),
    'count_frequent_purchase':list(most_frequent_cnt.values()),'rare_purchased':list(least_frequent.values())})
    most_purchased = [f'{i}_freq_confidence' for i in range(1,n+1)]
    # most_purchased[0] = 'item_order_cnt'
    often_rare_purchased = pd.merge(often_rare_purchased,products[['prod_name','product_type_name', 'product_group_name']].drop_duplicates(),on='prod_name',how='left')

    temp = pd.DataFrame(often_rare_purchased['count_frequent_purchase'].tolist(),columns=most_purchased) 
    often_rare_purchased = pd.concat([often_rare_purchased.drop('count_frequent_purchase',axis=1),temp],axis=1)

    for i in range(-1,-(n+1),-1):
        often_rare_purchased.iloc[:,i] = often_rare_purchased.iloc[:,i].values/often_rare_purchased.item_order_cnt.values
    often_rare_purchased['support'] = support[product_name]
    print(often_rare_purchased)
    for i in range(1,n+1):
        often_rare_purchased[f'{i}_freq_prod_lift'] = (temp[most_purchased[i-1]]/n_order)/(support[most_frequent[product_name][i-1]] * support[product_name])
    frequent_groups = [list(set(products[products['prod_name'] == often_rare_purchased.often_purchased[0][i]]['product_group_name']))[0] for i in range(n)]
    frequent_product_types = [list(set(products[products['prod_name'] == often_rare_purchased.often_purchased[0][i]]['product_type_name']))[0] for i in range(n)]
    often_rare_purchased['often_purchased_group'] = str(frequent_groups)
    often_rare_purchased['often_purchased_prod_types'] = str(frequent_product_types)
    return often_rare_purchased    
