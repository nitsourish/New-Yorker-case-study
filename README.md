# New-Yorker-case-study

## *Problem 1- Reporting*

### For detailed implementation please refer ./nbs/exploratory_report.ipynb 

### Report Design for Items Frequently and Rarely sold together

- A) May not be feassible or logical to consider all 105542 articles
- B) Article_id may not be suitable aggregator(eg. Strap top color black vs. Strap top color White as different articles) as multiple articles are too identical have comparable pattern.product_code level or above hierarchy report might be suitable.
- C) Flexibility- In terms of selection of hierarchy,filtering condition,no of Frequently and Rarely sold products(eg. for each item we may report any no of Frequently co-purchased items for any hierarchy) 
- D) Individual vs. All- Based on requirement reports for all products(total no vary based on fitering criteria) or customized for few products. Can be explored a particular product in detail.

### Instruction to generate Reports(both for all items and customized) 

- Clone the repository 

```bash

      git clone git@github.com:nitsourish/New-Yorker-case-study.git

```

- All inputs files are uploaded except for transactions_train.csv (Dataframe of the purchases each customer for each date with article id) because of size.Make sure to have it under ./data folder.

#### scripts 
- ./src/report_all.py
- ./src/individual_report.py

##### Please refer - List of CMD argparse

```bash
parser.add_argument('-m','--min_cnt', type = int, metavar = '',required=True,default=0,help='minimum order count for eligible article_id')
parser.add_argument('-p','--penetration', type = float, metavar = '',required=True,help='Fraction of unique customers for eligible article_id')
parser.add_argument('-ph','--product_hierarchy', type = str, metavar = '',required=True,help='product_hierarchy(article/product/product_type/product_group) of reporting')
parser.add_argument('-n','--num_prod', type = int, metavar = '',required=True, help='number of often/rarely purchased products for each products')
parser.add_argument('-all','--all', type = bool, metavar = '',required=False,default=True,help = 'boolean to indicate if report for all items in hierarchy')
parser.add_argument('-pl','--prods', type = list, metavar = '',required=False, default=[], help='list of items if all = False')
parser.add_argument('-prod','--product_name', type = str, metavar = '',required=True, help='name of the product for reporting')
```

##### Example script run from CMD

 - For any script to explore CMD argparse
 
 ```bash
 
 python report_all.py --help
 
 ```

- A) For all items

```bash

      python report_all.py -p 0.01 -ph 'prod_name' -n 3 -m 100

```

- B) For list of items

With same command line instruction, make following change in script:
  -- a) pass list of items in function argument prods with all = False(prepare_all_products_report(df,product_hierarchy = 'prod_name',n=3,all=False,prods = ['Jade HW Skinny Denim TRS','Tilda tank'])) 
  -- b) Alternatively can be provided as CMD argparse with defined format
  
 - C) For individual item detailed report
 
 ```bash

     python individual.py -p 0.01 -prod 'Perrie Slim Mom Denim TRS' -n 3 -m 100 > output.log

```

## *Problem 2: Interesting Problem - Product level Forecasting(of filtered products)*

### For detailed implementation please refer ./nbs/time_series.ipynb

### Design
 - A) Product specific Models
 - B) Unit of data- Daily level
 - C) With approx. 2 years of data, forecasting is for last 2 weeks(14 days) and model trained on rest of the data(for each product)
 - For quick implementation assume quarterly and additive seasonality and Polynomial trend of degree 1
 - I used TransformedTargetForecaster from sktime to build Forecasting pipeline using default XGBoost as regressor
 
 ### Assumption
  - A) Daily price will be available during forecasting period
  - B) Alternatively next day forecasting can be done based on last day price level

### Instruction to generate results 
- clone the repository(git@github.com:nitsourish/New-Yorker-case-study.git)
-  All inputs files are uploaded except for transactions_train.csv (Dataframe of the purchases each customer for each date with article id) because of size.Make sure to have it under ./data folder.
-  The productwise trained models also not uploaded.

#### scripts 
- ./src/forecasting_train_validation.py
- ./src/individual_report.py

##### Example script run from CMD

- A) For Model Data prep and model training-validation

 ```bash

    python forecasting_train_validation.py > output.log

```    

(To change filtering criteria for eligible product selection need to change penetration and or cum_sales_fraction arguments of function data_prep. For example to select more product reducee the penetration value to penetration=0.001,i.e. data_prep(transaction = train,products=products,penetration=0.001, cum_sales_fraction=0.0)

- B) For forecasting/inference

  As the models are not uploaded please run ``` forecasting_train_validation.py first ```

 ```bash

    python forecasting_inference.py > output.log

``` 
