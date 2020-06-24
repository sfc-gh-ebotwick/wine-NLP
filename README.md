# wine-NLP

Notebook originally authored by Elliott Botwick on Watson Studio Cloud.


## Project Overview & Motivation

The purpose of this project was to first apply data science best practices in building out a regression model predicting the quality of various wines, and then to bolster that model using Watson Natural Language Understanding (NLU) APIs. More background on the motivation in the project can be found in the Medium article write up - https://medium.com/@elliott.botwick_90525/end-to-end-wine-quality-modeling-with-a-boost-from-watson-nlp-services-8ebfd4137f6c?sk=3e471d765e570da1788678a26b7516e8


## Data

The data used in this analysis can be found on Kaggle's Wine Reviews project page: https://www.kaggle.com/zynicide/wine-reviews
The specific file used was *winemag-data_first150k.csv* This file contains wine reviews of 150,000 wines complete with their location, winery, variety, designation, price, points awarded, and a description field written by the reviewer. 

## Acknowledgements

Thank you to Stefan van der Stockt and Catherine Cao for their help in forming approaches and using NLU APIs. 

Also thank you to the authors of the following articles found online for their helpfulness in various areas:
https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
https://www.datacamp.com/community/tutorials/wordcloud-python

## Crisp-DM overview of Project

### Business Understanding 

Wine in the U.S. alone is a $70 billion dollar industry. Understanding the intricies of the differnces in what makes a good wine and what makes a great wine can be extremely profitable. Additionally, wine quality modeling serves as a proxy use case to demonstrate the value of IBM Watson NLU tooling that can be applicable in nearly any industry. 

### Data Understanding

A considerable amount of time was spent carrying out Exploratory Data Analysis (EDA). Findings varied from discovering a large portion of rows were duplciate records to understanding complex patterns in specific wine varieties and locations. Thorough EDA helped greatly in directing what data preperation steps were necessary to best model this data. 

### Prepare Data

A huge portion of this project was dedicated to finding answers to difficult data preperation tasks. This includes dealing with high cardinality cateogrical variables and imputing missing values in a hierarchical data pattern. The following sections detail the data preperation that went into this project:


#### Duplicate Rows:

One of the first steps I took in assessing the quality of this data was to inspect the proportion of unique values in each categorical variable. The results I found provided a good few challenges down the road. Here is a grape-themed visualization showing the breakdown of unique values in categorical feature:

./images/UniqueNonUniquePlot.png

The greater the sliver of green on the chart, the greater the number of values that category has

Naturally the description field makes sense as the most unique column here — there’s only so many countries or wine varieties in the world but essentially limitless permutations of an open text field. However, as can be seen above, the description field is still a long way from being 100% unique. This helped lead to the discovery that around 1/3 of this data were duplicate records.

I used the pandas drop_duplicates() function for a quick and easy solution to the replicated rows.

```wine.drop_duplicates(subset = ['description', 'winery', 'variety', 'country', 'province'],inplace = True)```

#### Missing Values

In the spirit of keeping a common format, I created a similar visualization for the proportion of null values in each column. Here are some more grapes:

../images/nullNonNull.png

Fortunately several of the fields were entirely populated, and some of the scarcely populated fields were found to have low associations with the target (points) were likely going to be dropped anyways.

Before explaining how missing values in region_1 were handled I’d like to clarify the location variable hierarchy a bit in case it may be unclear:

country > province > region_1 > region_2

So to deal with the missing values in region_1 I created a dict that contained each unique value of province and the highest occurring value of region_1 for that province. Then with a lambda function, the missing values in region_1 were imputed with the most common region in that province. You can see the code below:
```
regionDict = {}for p in wine.province.value_counts().index:
    if len(wine[wine.province ==p].region_1.value_counts()) < 1:
        regionDict.update({p: 'NoListedRegion'})
    else:
        regionDict.update({p:  wine[wine.province == p].region_1.value_counts().index[0]})
wine['region_1_imputed'] = wine[['province', 'region_1']].apply(lambda row: row.fillna(value = regionDict.get(row['province'])), axis = 1).loc[:, 'region_1']=
```
Some provinces didn’t have any listed regions, if this was the case the null value was replaced with ‘NoListedRegion’.

#### Dealing with Categorical Variables with high cardinality

This ended up being the area of data prep I probably spent the most time on — partially because it was difficult and partially because it was interesting. With some help from a few very clever friends and colleagues, I came up with a methodology I believe worked out well.

For categorical variables with > 100 levels, the first N levels to make up 95% of all values in the column were identified and all other levels were cast to ‘other’. Those columns were then one-hot encoded. If it took more than 100 levels to make up 95% of all values in the column I decided to either take the first 100 values and one-hot encode or just to index the column (I know, I know) on a case-by-case basis.
A screenshot of the notebook showing how we found that the 98 most popular varieties make up 95% of all varieties

nbscreenshot.png

Some features, such as winery, needed over 1000 levels to make up 95% of all values. That column was simply indexed here but in future analysis an approach that may be worth taking is doing some clustering of those columns to form logical grouping of winery values and reduce dimensionality by replacing the original value with a cluster value.

At this point we did have our data ready to train some models and created a gradient boosted model with an Adjusted R² value of 0.586 and a MSE of 4.29. Not bad but lets see what we can do to improve!

### Data Modeling

This use case required modeling a continuous target (points given in a wine review) on a scale of [0, 100] modeling didn't actually represent a huge challenge in this use case. Two series of model training took place, one with the base set of features and one with added features representing the emotional breakdown of the description. For each round of model training a linear regression and a gradient boosted model were fitted. The gradient boosted model was superior in each case and the model with the emotion features outperformed the models without those features. 

### Evaluating Results

Mean Squared Error (MSE) and R^2 were the two primary metrics used to evaluate the models created in this project. 

For the baseline model (no NLP) the performance metrics on the test set were as follows:
Linear Regression: MSE - 4.354; R^2 - 0.580
Gradient Boosted Regressor: MSE - 4.290; R^2 - 0.587

For the enhanced model (with NLP) the performance metrics on the test set were as follows:
Linear Regression: MSE - 4.146; R^2 - 0.601
Gradient Boosted Regressor : MSE - **4.103**; R^2 - **0.605**

The gradient boosted regressor trained on the enhanced data with NLP applied showed as the strongest model in this case. 

## Packages used 

The development environment used was Python 3.6 in a Jupyter Notebook on Watson Studio cloud. The pip list from the notebook is below:

Package                            Version   
---------------------------------- ----------
absl-py                            0.7.0     
alabaster                          0.7.12    
anaconda-client                    1.7.2     
anaconda-project                   0.8.2     
arcgis                             1.6.0     
asn1crypto                         0.24.0    
astor                              0.7.1     
astroid                            2.1.0     
astropy                            3.1.1     
astunparse                         1.6.2     
atomicwrites                       1.3.0     
attrs                              18.2.0    
autoai-libs                        1.10.5    
Babel                              2.6.0     
backcall                           0.1.0     
backports.os                       0.1.1     
backports.shutil-get-terminal-size 1.0.0     
beautifulsoup4                     4.7.1     
biopython                          1.72      
bitarray                           0.8.3     
bkcharts                           0.2       
blaze                              0.11.3    
bleach                             3.1.0     
bokeh                              1.0.4     
boto                               2.49.0    
boto3                              1.9.82    
botocore                           1.12.82   
Bottleneck                         1.2.1     
brunel                             2.3       
ca-data-connector                  11.1.7    
category-encoders                  2.0.0     
certifi                            2020.4.5.1
cffi                               1.11.5    
chardet                            3.0.4     
Click                              7.0       
cloudpickle                        0.7.0     
clyent                             1.2.2     
colorama                           0.4.1     
colour                             0.1.5     
contextlib2                        0.5.5     
cplex                              12.10.0.1 
cryptography                       2.5       
cx-Oracle                          7.0.0     
cycler                             0.10.0    
Cython                             0.29.5    
cytoolz                            0.9.0.1   
dask                               1.1.1     
datashape                          0.5.4     
decorator                          4.3.2     
defusedxml                         0.5.0     
dill                               0.2.8.2   
distributed                        1.25.3    
docloud                            1.0.375   
docplex                            2.14.186  
docutils                           0.14      
entrypoints                        0.3       
et-xmlfile                         1.0.1     
fastcache                          1.0.2     
filelock                           3.0.10    
Flask                              1.0.2     
Flask-Cors                         3.0.7     
future                             0.17.1    
gast                               0.2.2     
geographiclib                      1.49      
geojson                            2.4.1     
geopy                              1.18.1    
gevent                             1.4.0     
glob2                              0.6       
gmpy2                              2.0.8     
greenlet                           0.4.15    
grpcio                             1.16.1    
h5py                               2.9.0     
heapdict                           1.0.0     
html5lib                           1.0.1     
ibm-bias-detection                 1.0.8     
ibm-cos-sdk                        2.4.3     
ibm-cos-sdk-core                   2.4.3     
ibm-cos-sdk-s3transfer             2.4.3     
ibm-db                             2.0.9     
ibm-db-sa                          0.3.4     
ibmdbpy                            0.1.5     
idna                               2.8       
imageio                            2.4.1     
imagesize                          1.1.0     
importlib-metadata                 0.7       
ipykernel                          5.1.0     
ipython                            7.2.0     
ipython-genutils                   0.2.0     
ipywidgets                         7.4.2     
isort                              4.3.4     
itsdangerous                       1.1.0     
JayDeBeApi                         1.1.1     
jdcal                              1.4       
jedi                               0.13.2    
jeepney                            0.4       
Jinja2                             2.10      
jmespath                           0.9.3     
JPype1                             0.6.3     
JPype1-py3                         0.5.5.2   
jsonschema                         2.6.0     
jupyter                            1.0.0     
jupyter-client                     5.2.4     
jupyter-console                    6.0.0     
jupyter-core                       4.4.0     
jupyter-pip                        0.3.1     
jupyterlab                         0.35.3    
jupyterlab-server                  0.2.0     
Keras                              2.2.4     
Keras-Applications                 1.0.6     
Keras-Preprocessing                1.0.5     
keyring                            18.0.0    
kiwisolver                         1.0.1     
lazy                               1.4       
lazy-object-proxy                  1.3.1     
libarchive-c                       2.8       
lief                               0.9.0     
llvmlite                           0.27.0    
locket                             0.2.0     
lomond                             0.3.3     
lxml                               4.3.1     
Markdown                           3.0.1     
MarkupSafe                         1.1.0     
matplotlib                         3.0.2     
mccabe                             0.6.1     
mistune                            0.8.4     
mkl-fft                            1.0.10    
mkl-random                         1.0.2     
mock                               2.0.0     
more-itertools                     5.0.0     
mpld3                              0.3       
mpmath                             1.1.0     
msgpack                            0.6.1     
multipledispatch                   0.6.0     
nbconvert                          5.4.0     
nbformat                           4.4.0     
networkx                           2.2       
nltk                               3.4       
nose                               1.3.7     
notebook                           5.7.8     
numba                              0.42.0    
numexpr                            2.6.9     
numpy                              1.15.4    
numpydoc                           0.8.0     
odo                                0.5.1     
olefile                            0.46      
openpyxl                           2.6.0     
packaging                          19.0      
pandas                             0.24.1    
pandocfilters                      1.4.2     
parso                              0.3.2     
partd                              0.3.9     
path.py                            11.5.0    
pathlib2                           2.3.3     
patsy                              0.5.1     
pbr                                5.1.3     
pep8                               1.7.1     
pexpect                            4.6.0     
pickleshare                        0.7.5     
Pillow                             5.4.1     
pip                                19.1.1    
pixiedust                          1.1.17    
pkginfo                            1.5.0.1   
plotly                             3.6.1     
pluggy                             0.8.1     
ply                                3.11      
project-lib                        2.0.0     
prometheus-client                  0.5.0     
prompt-toolkit                     2.0.8     
protobuf                           3.6.1     
psutil                             5.5.0     
psycopg2                           2.7.6.1   
ptyprocess                         0.6.0     
py                                 1.7.0     
pyarrow                            0.11.1    
pycodestyle                        2.5.0     
pycosat                            0.6.3     
pycparser                          2.19      
pycrypto                           2.6.1     
pycurl                             7.43.0.2  
pyflakes                           2.1.0     
Pygments                           2.3.1     
pylint                             2.2.2     
pymssql                            2.1.4     
pyodbc                             4.0.25    
pyOpenSSL                          19.0.0    
pyparsing                          2.3.1     
pypyodbc                           1.3.4     
pyshp                              2.1.0     
PySocks                            1.6.8     
pytest                             4.2.1     
pytest-arraydiff                   0.3       
pytest-astropy                     0.5.0     
pytest-doctestplus                 0.2.0     
pytest-openfiles                   0.3.2     
pytest-remotedata                  0.3.1     
python-dateutil                    2.7.5     
pytz                               2018.9    
PyWavelets                         1.0.1     
PyYAML                             3.13      
pyzmq                              17.1.2    
QtAwesome                          0.5.6     
qtconsole                          4.4.3     
QtPy                               1.6.0     
requests                           2.21.0    
retrying                           1.3.3     
rope                               0.11.0    
ruamel-yaml                        0.15.46   
s3transfer                         0.1.13    
scikit-image                       0.14.1    
scikit-learn                       0.20.3    
scipy                              1.2.0     
seaborn                            0.9.0     
SecretStorage                      3.1.0     
Send2Trash                         1.5.0     
setuptools                         40.8.0    
simplegeneric                      0.8.1     
singledispatch                     3.4.0.3   
six                                1.12.0    
snowballstemmer                    1.2.1     
sortedcollections                  1.1.2     
sortedcontainers                   2.1.0     
soupsieve                          1.7.1     
Sphinx                             1.8.4     
sphinxcontrib-websupport           1.1.0     
spyder                             3.3.3     
spyder-kernels                     0.4.2     
SQLAlchemy                         1.2.18    
statsmodels                        0.9.0     
streamsx                           1.13.14   
sympy                              1.3       
tables                             3.4.4     
tabulate                           0.8.2     
tblib                              1.3.2     
tensorflow                         1.13.1    
tensorflow-estimator               1.13.0    
termcolor                          1.1.0     
terminado                          0.8.1     
testpath                           0.4.2     
toolz                              0.9.0     
tornado                            5.1.1     
tqdm                               4.31.1    
traitlets                          4.3.2     
typed-ast                          1.3.1     
unicodecsv                         0.14.1    
urllib3                            1.24.1    
watson-machine-learning-client     1.0.376   
wcwidth                            0.1.7     
webencodings                       0.5.1     
Werkzeug                           0.14.1    
wheel                              0.32.3    
widgetsnbextension                 3.4.2     
wrapt                              1.11.1    
wurlitzer                          1.0.2     
xlrd                               1.2.0     
XlsxWriter                         1.1.2     
xlwt                               1.3.0     
zict                               0.1.3     
