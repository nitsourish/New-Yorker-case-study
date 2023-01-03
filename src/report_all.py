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

logger = logging.getLogger()

data_path = "../data/"
report_path = "../data/reports/"

parser = argparse.ArgumentParser(
    description="reports often or rarely sold together items"
)
parser.add_argument(
    "-m",
    "--min_cnt",
    type=int,
    metavar="",
    required=True,
    default=0,
    help="minimum order count for eligible article_id",
)
parser.add_argument(
    "-p",
    "--penetration",
    type=float,
    metavar="",
    required=True,
    help="Fraction of unique customers for eligible article_id",
)
parser.add_argument(
    "-ph",
    "--product_hierarchy",
    type=str,
    metavar="",
    required=True,
    help="product_hierarchy(article/product/product_type/product_group) of reporting",
)
parser.add_argument(
    "-n",
    "--num_prod",
    type=int,
    metavar="",
    required=True,
    help="number of often/rarely purchased products for each products",
)
parser.add_argument(
    "-all",
    "--all",
    type=bool,
    metavar="",
    required=False,
    default=True,
    help="boolean to indicate if report for all items in hierarchy",
)
parser.add_argument(
    "-pl",
    "--prods",
    type=list,
    metavar="",
    required=False,
    default=[],
    help="list of items if all = False",
)
args = parser.parse_args()


def data_prep(
    transaction: pd.DataFrame,
    products: pd.DataFrame,
    penetration: float,
    min_cnt: int = 1,
):

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
    # For Memory optimization
    transaction["customer_id"] = (
        transaction["customer_id"].apply(lambda x: int(x[-16:], 16)).astype("int64")
    )
    transaction["article_id"] = transaction.article_id.astype("int32")
    products_col = [
        "article_id",
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
    ]
    df = pd.merge(
        transaction[["t_dat", "customer_id", "article_id"]],
        products[products_col],
        on=["article_id"],
        how="left",
    )
    cnt = pd.DataFrame(df.article_id.value_counts())
    cnt.columns = ["counts"]
    if penetration > 0:
        custn = transaction.groupby("article_id")["customer_id"].nunique()
        cust = transaction["customer_id"].nunique()
        custn = custn / cust
        custn = custn.sort_values(ascending=False)
        prod_index_cutoff = list(custn[custn >= penetration].index)
        df = df[df.article_id.isin(prod_index_cutoff)]
    if min_cnt > 1:
        cnt = transaction.article_id.value_counts()
        prod_index_cutoff = list(cnt[cnt >= min_cnt].index)
        df = df[df.article_id.isin(prod_index_cutoff)]
    df["order_instance"] = df["customer_id"].astype(str) + df["t_dat"]
    return df


def prepare_all_products_report(
    df: pd.DataFrame,
    product_hierarchy: str,
    num_prod: int,
    all: bool = True,
    prods: list = [],
):

    """
    Reporting function to generate and saving dataframe of frequently and rarely product combination along with metrics.

    Args:
        params:
            df: order instance wise Merged and filtered Dataframe with eligible article/products
            product_hierarchy: product_hierarchy(article/product/product_type/product_group) of reporting
            num_prod: number of often/rarely purchased products for each products
            all: boolean to indicate if report for all items in hierarchy, default is True
            prods:optional list of items for customized report only for these items.To use this all indicator should be False

    Returns:
        a) Report of list(based on n) of frequently/rarely co-purchased items for each item
        along with support for respective item and confidence metrics for each frequently co-purchased items.
        b) support(order count for individual item/total order count) for each item w.r.t. product_hierarchy input
    """
    cnt = df[product_hierarchy].value_counts()
    if not all:
        try:
            cnt = cnt[cnt.index.isin(prods)]
        except:
            raise ValueError("item list is empty")
    n_order = df["order_instance"].nunique()
    order_cnt = df.groupby([product_hierarchy])["order_instance"].nunique()
    support = order_cnt / n_order
    order_cnt = order_cnt.to_dict()

    most_frequent = {}
    most_frequent_cnt = {}
    least_frequent = {}
    item_order_cnt = {}
    item_support = {}
    # leat_frequent_cnt = {}
    for j, i in tqdm(enumerate(cnt.index.values)):
        orders = df.loc[df[product_hierarchy] == i, "order_instance"].unique()
        cnt2 = df.loc[
            (df.order_instance.isin(orders)) & (df[product_hierarchy] != i),
            product_hierarchy,
        ].value_counts()
        most_frequent[i] = [cnt2.index[j] for j in range(1, num_prod + 1)]
        least_frequent[i] = [cnt2.index[j] for j in range(-1, -(num_prod + 1), -1)]
        most_frequent_cnt[i] = [cnt2.values[j] for j in range(1, num_prod + 1)]
        item_order_cnt[i] = order_cnt[i]
        item_support[i] = support.to_dict()[i]
        # leat_frequent_cnt[i.item()] = [cnt2.values[-1], cnt2.values[-2], cnt2.values[-3]]
    often_rare_purchased = pd.DataFrame(
        {
            "items": list(most_frequent.keys()),
            "item_order_cnt": item_order_cnt.values(),
            "often_purchased": list(most_frequent.values()),
            "count_frequent_purchase": list(most_frequent_cnt.values()),
            "rare_purchased": list(least_frequent.values()),
            "support": item_support.values(),
        }
    )
    most_purchased = [f"{i}_freq_confidence" for i in range(1, num_prod + 1)]
    temp = pd.DataFrame(
        often_rare_purchased["count_frequent_purchase"].tolist(), columns=most_purchased
    )
    often_rare_purchased = pd.concat(
        [often_rare_purchased.drop("count_frequent_purchase", axis=1), temp], axis=1
    )
    order_cnt
    for i in range(-1, -(num_prod + 1), -1):
        often_rare_purchased.iloc[:, i] = (
            often_rare_purchased.iloc[:, i].values
            / often_rare_purchased.item_order_cnt.values
        )
    often_rare_purchased = often_rare_purchased.sort_values("support", ascending=False)
    support = support.to_dict()
    return often_rare_purchased, support


if __name__ == "__main__":

    logging.basicConfig(filename="output.log", level=logging.DEBUG)
    logger.addHandler(logging.FileHandler("output.log", "a"))
    print(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"))
    train = pd.read_csv(os.path.join(data_path, "transactions_train.csv.zip"))
    products = pd.read_csv(os.path.join(data_path, "articles.csv.zip"))
    print("Data Read")

    df = data_prep(
        transaction=train, products=products, min_cnt=0, penetration=args.penetration
    )
    print(f"Data prepared {df.shape}")

    often_rare_purchased, support = prepare_all_products_report(
        df,
        product_hierarchy=args.product_hierarchy,
        num_prod=args.num_prod,
        all=args.all,
        prods=args.prods,
    )
    print(often_rare_purchased.head())
    often_rare_purchased.to_csv(
        os.path.join(report_path, "prod_name_wise_report_adhoc.csv"), index=False
    )
