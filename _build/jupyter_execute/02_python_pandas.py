#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Starting-pandas" data-toc-modified-id="Starting-pandas-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Starting pandas</a></span></li><li><span><a href="#Data-import-and-export" data-toc-modified-id="Data-import-and-export-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data import and export</a></span></li><li><span><a href="#Exploring-a-data-set" data-toc-modified-id="Exploring-a-data-set-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Exploring a data set</a></span></li><li><span><a href="#Data-structures-and-types" data-toc-modified-id="Data-structures-and-types-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Data structures and types</a></span><ul class="toc-item"><li><span><a href="#pandas.Series" data-toc-modified-id="pandas.Series-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>pandas.Series</a></span></li><li><span><a href="#pandas.DataFrame" data-toc-modified-id="pandas.DataFrame-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>pandas.DataFrame</a></span><ul class="toc-item"><li><span><a href="#Creating-a-DataFrame" data-toc-modified-id="Creating-a-DataFrame-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Creating a DataFrame</a></span></li><li><span><a href="#Working-with-a-DataFrame" data-toc-modified-id="Working-with-a-DataFrame-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>Working with a DataFrame</a></span></li><li><span><a href="#Extracting-rows-and-columns" data-toc-modified-id="Extracting-rows-and-columns-5.2.3"><span class="toc-item-num">5.2.3&nbsp;&nbsp;</span>Extracting rows and columns</a></span></li><li><span><a href="#Boolean-selection" data-toc-modified-id="Boolean-selection-5.2.4"><span class="toc-item-num">5.2.4&nbsp;&nbsp;</span>Boolean selection</a></span></li><li><span><a href="#query" data-toc-modified-id="query-5.2.5"><span class="toc-item-num">5.2.5&nbsp;&nbsp;</span><code>query</code></a></span></li><li><span><a href="#Replacing-values-in-a-DataFrame" data-toc-modified-id="Replacing-values-in-a-DataFrame-5.2.6"><span class="toc-item-num">5.2.6&nbsp;&nbsp;</span>Replacing values in a DataFrame</a></span></li></ul></li><li><span><a href="#Categorical-data" data-toc-modified-id="Categorical-data-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Categorical data</a></span><ul class="toc-item"><li><span><a href="#Re-organizing-categories" data-toc-modified-id="Re-organizing-categories-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>Re-organizing categories</a></span></li></ul></li><li><span><a href="#Missing-data" data-toc-modified-id="Missing-data-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Missing data</a></span></li></ul></li><li><span><a href="#Data-transformation" data-toc-modified-id="Data-transformation-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Data transformation</a></span><ul class="toc-item"><li><span><a href="#Arithmetic-operations" data-toc-modified-id="Arithmetic-operations-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Arithmetic operations</a></span></li><li><span><a href="#Concatenation-of-data-sets" data-toc-modified-id="Concatenation-of-data-sets-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Concatenation of data sets</a></span><ul class="toc-item"><li><span><a href="#Adding-columns" data-toc-modified-id="Adding-columns-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>Adding columns</a></span></li></ul></li><li><span><a href="#Merging-data-sets" data-toc-modified-id="Merging-data-sets-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Merging data sets</a></span></li><li><span><a href="#Tidy-data-principles-and-reshaping-datasets" data-toc-modified-id="Tidy-data-principles-and-reshaping-datasets-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Tidy data principles and reshaping datasets</a></span></li><li><span><a href="#Melting-(unpivoting)-data" data-toc-modified-id="Melting-(unpivoting)-data-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Melting (unpivoting) data</a></span></li><li><span><a href="#Separating-columns-containing-multiple-variables" data-toc-modified-id="Separating-columns-containing-multiple-variables-6.6"><span class="toc-item-num">6.6&nbsp;&nbsp;</span>Separating columns containing multiple variables</a></span></li><li><span><a href="#Pivot/spread-datasets" data-toc-modified-id="Pivot/spread-datasets-6.7"><span class="toc-item-num">6.7&nbsp;&nbsp;</span>Pivot/spread datasets</a></span></li></ul></li><li><span><a href="#Data-aggregation-and-split-apply-combine" data-toc-modified-id="Data-aggregation-and-split-apply-combine-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Data aggregation and split-apply-combine</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Transformation" data-toc-modified-id="Transformation-7.0.1"><span class="toc-item-num">7.0.1&nbsp;&nbsp;</span>Transformation</a></span></li><li><span><a href="#Filter" data-toc-modified-id="Filter-7.0.2"><span class="toc-item-num">7.0.2&nbsp;&nbsp;</span>Filter</a></span></li></ul></li></ul></li></ul></div>
%%R
reticulate::use_condaenv('ds', required=T)
# # Pandas
# 
# ## Introduction
# 
# `pandas` is the Python Data Analysis package. It allows for data ingestion, transformation and cleaning, and creates objects that can then be passed on to analytic packages like `statsmodels` and `scikit-learn` for modeling and packages like `matplotlib`, `seaborn`, and `plotly` for visualization. 
# 
# `pandas` is built on top of numpy, so many numpy functions are commonly used in manipulating `pandas` objects. 
# 
# > `pandas` is a pretty extensive package, and we'll only be able to cover some of its features. For more details, there is free online documentation at [pandas.pydata.org](https://pandas.pydata.org). You can also look at the book ["Python for Data Analysis (2nd edition)"](https://www.amazon.com/Python-Data-Analysis-Wrangling-IPython-dp-1491957662/dp/1491957662/) by Wes McKinney, the original developer of the pandas package, for more details.
# 
# ## Starting pandas
# 
# As with any Python module, you have to "activate" `pandas` by using `import`. The "standard" alias for `pandas` is `pd`. We will also import `numpy`, since `pandas` uses some `numpy` functions in the workflows. 

# In[1]:


import numpy as np
import pandas as pd


# ## Data import and export
# 
# Most data sets you will work with are set up in tables, so are rectangular in shape. Think Excel spreadsheets. In `pandas` the structure that will hold this kind of data is a `DataFrame`.  We can read external data into a `DataFrame` using one of many `read_*` functions. We can also write from a `DataFrame` to a variety of formats using `to_*` functions. The most common of these are listed below:
# 
# | Format type | Description | reader       | writer     |
# | ----------- | ----------- | ------------ | ---------- |
# | text        | CSV         | read_csv     | to_csv     |
# |             | Excel       | read_excel   | to_excel   |
# | text        | JSON        | read_json    | to_json    |
# | binary      | Feather     | read_feather | to_feather |
# | binary      | SAS         | read_sas     |            |
# | SQL         | SQL         | read_sql     | to_sql     |
# 
# We'll start by reading in the `mtcars` dataset stored as a CSV file

# In[2]:


pd.read_csv('data/mtcars.csv')


# This just prints out the data, but then it's lost. To use this data, we have to give it a name, so it's stored in Python's memory

# In[3]:


mtcars = pd.read_csv('data/mtcars.csv')


# > One of the big differences between a spreadsheet program and a programming language from the data science perspective is that you have to load data into the programming language. It's not "just there" like Excel. This is a good thing, since it allows the common functionality of the programming language to work across multiple data sets, and also keeps the original data set pristine. Excel users can run into problems and [corrupt their data](https://nature.berkeley.edu/garbelottoat/?p=1488) if they are not careful.
# 
# If we wanted to write this data set back out into an Excel file, say, we could do

# In[4]:


mtcars.to_excel('data/mtcars.xlsx')


# > You may get an error if you don't have the `openpyxl` package installed. You can easily install it from the Anaconda prompt using `conda install openpyxl` and following the prompts. 
# 

# ## Exploring a data set
# 
# We would like to get some idea about this data set. There are a bunch of functions linked to the `DataFrame` object that help us in this. First we will use `head` to see the first 8 rows of this data set

# In[5]:


mtcars.head(8)


# This is our first look into this data. We notice a few things. Each column has a name, and each row has an *index*, starting at 0. 
# 
# > If you're interested in the last N rows, there is a corresponding `tail` function
# 
# Let's look at the data types of each of the columns

# In[6]:


mtcars.dtypes


# This tells us that some of the variables, like `mpg` and `disp`, are floating point (decimal) numbers, several are integers, and `make` is an "object". The `dtypes` function borrows from `numpy`, where there isn't really a type for character or categorical variables. So most often, when you see "object" in the output of `dtypes`, you think it's a character or categorical variable. 
# 
# We can also look at the data structure in a bit more detail.

# In[7]:


mtcars.info()


# This tells us that this is indeed a `DataFrame`, wth 12 columns, each with 32 valid observations. Each row has an index value ranging from 0 to 11. We also get the approximate size of this object in memory.
# 
# You can also quickly find the number of rows and columns of a data set by using `shape`, which is borrowed from numpy.

# In[8]:


mtcars.shape


# More generally, we can get a summary of each variable using the `describe` function

# In[9]:


mtcars.describe()


# These are usually the first steps in exploring the data.

# ## Data structures and types
# 
# pandas has two main data types: `Series` and `DataFrame`. These are analogous to vectors and matrices, in that a `Series` is 1-dimensional while a `DataFrame` is 2-dimensional. 
# 
# ### pandas.Series
# 
# The `Series` object holds data from a single input variable, and is required, much like numpy arrays, to be homogeneous in type. You can create `Series` objects from lists or numpy arrays quite easily

# In[10]:


s = pd.Series([1,3,5,np.nan, 9, 13])
s


# In[11]:


s2 = pd.Series(np.arange(1,20))
s2


# You can access elements of a `Series` much like a `dict`

# In[12]:


s2[4]


# There is no requirement that the index of a `Series` has to be numeric. It can be any kind of scalar object

# In[13]:


s3 = pd.Series(np.random.normal(0,1, (5,)), index = ['a','b','c','d','e'])
s3


# In[14]:


s3['d']


# In[15]:


s3['a':'d']


# Well, slicing worked, but it gave us something different than expected. It gave us both the start **and** end of the slice, which is unlike what we've encountered so far!! 
# 
# It turns out that in `pandas`, slicing by index actually does this. It is a discrepancy from `numpy` and Python in general that we have to be careful about. 

# You can extract the actual values into a numpy array

# In[16]:


s3.to_numpy()


# In fact, you'll see that much of `pandas`' structures are build on top of `numpy` arrays. This is a good thing, since you can take advantage of the powerful numpy functions that are built for fast, efficient scientific computing. 

# Making the point about slicing again, 

# In[17]:


s3.to_numpy()[0:3]


# This is different from index-based slicing done earlier.

# ### pandas.DataFrame
# 
# The `DataFrame` object holds a rectangular data set. Each column of a `DataFrame` is a `Series` object. This means that each column of a `DataFrame` must be comprised of data of the same type, but different columns can hold data of different types. This structure is extremely useful in practical data science. The invention of this structure was, in my opinion, transformative in making Python an effective data science tool.

# #### Creating a DataFrame
# 
# The `DataFrame` can be created by importing data, as we saw in the previous section. It can also be created by a few methods within Python.
# 
# First, it can be created from a 2-dimensional `numpy` array.

# In[18]:


rng = np.random.RandomState(25)
d1 = pd.DataFrame(rng.normal(0,1, (4,5)))
d1


# You will notice that it creates default column names, that are merely the column number, starting from 0. We can also create the column names and row index (similar to the `Series` index we saw earlier) directly during creation.

# In[19]:


d2 = pd.DataFrame(rng.normal(0,1, (4, 5)), 
                  columns = ['A','B','C','D','E'], 
                  index = ['a','b','c','d'])
d2


# > We could also create a `DataFrame` from a list of lists, as long as things line up, just as we showed for `numpy` arrays. However, to me, other ways, including the `dict` method below, make more sense.

# We can change the column names (which can be extracted and replaced with the `columns` attribute) and the index values (using the `index` attribute).

# In[20]:


d2.columns


# In[21]:


d2.columns = pd.Index(['V'+str(i) for i in range(1,6)]) # Index creates the right objects for both column names and row names, which can be extracted and changed with the `index` attribute
d2


# **Exercise:** Can you explain what I did in the list comprehension above? The key points are understanding `str` and how I constructed the `range`.

# In[22]:


d2.index = ['o1','o2','o3','o4']
d2


# You can also extract data from a homogeneous `DataFrame` to a `numpy` array

# In[23]:


d1.to_numpy()


# > It turns out that you can use `to_numpy` for a non-homogeneous `DataFrame` as well. `numpy` just makes it homogeneous by assigning each column the data type `object`. This also limits what you can do in `numpy` with the array and may require changing data types using the [`astype` function](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html). There is some more detail about the `object` data type in the Python Tools for Data Science ([notebook](01_python_tools_ds.ipynb#object), [PDF](01_python_tools_ds.pdf)) document.

# The other easy way to create a `DataFrame` is from a `dict` object, where each component object is either a list or a numpy array, and is homogeneous in type. One exception is if a component is of size 1; then it is repeated to meet the needs of the `DataFrame`'s dimensions

# In[24]:


df = pd.DataFrame({
    'A':3.,
    'B':rng.random_sample(5),
    'C': pd.Timestamp('20200512'),
    'D': np.array([6] * 5),
    'E': pd.Categorical(['yes','no','no','yes','no']),
    'F': 'NIH'})
df


# In[25]:


df.info()


# We note that C is a date object, E is a category object, and F is a text/string object. pandas has excellent time series capabilities (having origins in FinTech), and the `TimeStamp` function creates datetime objects which can be queried and manipulated in Python. We'll describe category data in the next section.

# You can also create a `DataFrame` where each column is composed of composite objects, like lists and dicts, as well. This might have limited value in some settings, but may be useful in others. In particular, this allows capabilities like the [*list-column* construct in R tibbles](https://jennybc.github.io/purrr-tutorial/ls13_list-columns.html). For example, 

# In[26]:


pd.DataFrame({'list' :[[1,2],[3,4],[5,6]],
             'tuple' : [('a','b'), ('c','d'), ('e','f')],
              'set' : [{'A','B','C'}, {'D','E'}, {'F'}], 
            'dicts' : [{'A': [1,2,3]}, {'B':[5,6,8]}, {'C': [3,9]}]})


# #### Working with a DataFrame
# 
# You can extract particular columns of a `DataFrame` by name

# In[27]:


df['E']


# In[28]:


df['B']


# > There is also a shortcut for accessing single columns, using Python's dot (`.`) notation.

# In[29]:


df.B


# > This notation can be more convenient if we need to perform operations on a single column. If we want to extract multiple columns, this notation will not work. Also, if we want to create new columns or replace existing columns, we need to use the array notation with the column name in quotes. 

# Let's look at slicing a `DataFrame`
# 
# #### Extracting rows and columns
# 
# There are two extractor functions in `pandas`:
# 
# + `loc` extracts by label (index label, column label, slice of labels, etc.
# + `iloc` extracts by index (integers, slice objects, etc.
# 

# In[30]:


df2 = pd.DataFrame(rng.randint(0,10, (5,4)), 
                  index = ['a','b','c','d','e'],
                  columns = ['one','two','three','four'])
df2


# First, let's see what naively slicing this `DataFrame` does.

# In[31]:


df2['one']


# Ok, that works. It grabs one column from the dataset. How about the dot notation?

# In[32]:


df2.one


# Let's see what this produces.

# In[33]:


type(df2.one)


# So this is a series, so we can potentially do slicing of this series.

# In[34]:


df2.one['b']


# In[35]:


df2.one['b':'d']


# In[36]:


df2.one[:3]


# Ok, so we have all the `Series` slicing available. The problem here is in semantics, in that we are grabbing one column and then slicing the rows. That doesn't quite work with our sense that a `DataFrame` is a rectangle with rows and columns, and we tend to think of rows, then columns. 
# 
# Let's see if we can do column slicing with this. 

# In[37]:


df2[:'two']


# That's not what we want, of course. It's giving back the entire data frame. We'll come back to this.

# In[38]:


df2[['one','three']]


# That works correctly though. We can give a list of column names. Ok. 
# 
# How about row slices?

# In[39]:


#df2['a'] # Doesn't work
df2['a':'c'] 


# Ok, that works. It slices rows, but includes the largest index, like a `Series` but unlike `numpy` arrays. 

# In[40]:


df2[0:2]


# Slices by location work too, but use the `numpy` slicing rules. 

# This entire extraction method becomes confusing. Let's simplify things for this, and then move on to more consistent ways to extract elements of a `DataFrame`. Let's agree on two things. If we're going the direct extraction route, 
# 
# 1. We will extract single columns of a `DataFrame` with `[]` or `.`, i.e., `df2['one']` or `df2.one`
# 1. We will extract slices of rows of a `DataFrame` using location only, i.e., `df2[:3]`. 
# 
# For everything else, we'll use two functions, `loc` and `iloc`.
# 
# + `loc` extracts elements like a matrix, using index and columns
# + `iloc` extracts elements like a matrix, using location

# In[41]:


df2.loc[:,'one':'three']


# In[42]:


df2.loc['a':'d',:]


# In[43]:


df2.loc['b', 'three']


# So `loc` works just like a matrix, but with `pandas` slicing rules (include largest index)

# In[44]:


df2.iloc[:,1:4]


# In[45]:


df2.iloc[1:3,:]


# In[46]:


df2.iloc[1:3, 1:4]


# `iloc` slices like a matrix, but uses `numpy` slicing conventions (does **not** include highest index)
# 
# If we want to extract a single element from a dataset, there are two functions available, `iat` and `at`, with behavior corresponding to `iloc` and `loc`, respectively.

# In[47]:


df2.iat[2,3]


# In[48]:


df2.at['b','three']


# #### Boolean selection
# 
# We can also use tests to extract data from a `DataFrame`. For example, we can extract only rows where column labeled `one` is greater than 3. 

# In[49]:


df2[df2.one > 3]


# We can also do composite tests. Here we ask for rows where `one` is greater than 3 and `three` is less than 9

# In[50]:


df2[(df2.one > 3) & (df2.three < 9)]


# #### `query`
# 
# `DataFrame`'s have a `query` method allowing selection using a Python expression

# In[51]:


n = 10
df = pd.DataFrame(np.random.rand(n, 3), columns = list('abc'))
df


# In[52]:


df[(df.a < df.b) & (df.b < df.c)]


# We can equivalently write this query as 

# In[53]:


df.query('(a < b) & (b < c)')


# #### Replacing values in a DataFrame
# 
# We can replace values within a DataFrame either by position or using a query. 

# In[54]:


df2


# In[55]:


df2['one'] = [2,5,2,5,2]
df2


# In[56]:


df2.iat[2,3] = -9 # missing value
df2


# Let's now replace values using `replace` which is more flexible. 

# In[57]:


df2.replace(0, -9) # replace 0 with -9


# In[58]:


df2.replace({2: 2.5, 8: 6.5}) # multiple replacements


# In[59]:


df2.replace({'one': {5: 500}, 'three': {0: -9, 8: 800}}) 
# different replacements in different columns


# See more examples in the [documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html?highlight=replace#pandas.DataFrame.replace)

# ### Categorical data
# 
# `pandas` provides a `Categorical` function and a `category` object type to Python. This type is analogous to the `factor` data type in R. It is meant to address categorical or discrete variables, where we need to use them in analyses. Categorical variables typically take on a small number of unique values, like gender, blood type, country of origin, race, etc. 
# 
# You can create categorical `Series` in a couple of ways:

# In[60]:


s = pd.Series(['a','b','c'], dtype='category')


# In[61]:


df = pd.DataFrame({
    'A':3.,
    'B':rng.random_sample(5),
    'C': pd.Timestamp('20200512'),
    'D': np.array([6] * 5),
    'E': pd.Categorical(['yes','no','no','yes','no']),
    'F': 'NIH'})
df['F'].astype('category')


# You can also create `DataFrame`'s where each column is categorical

# In[62]:


df = pd.DataFrame({'A': list('abcd'), 'B': list('bdca')})
df_cat = df.astype('category')
df_cat.dtypes


# You can explore categorical data in a variety of ways

# In[63]:


df_cat['A'].describe()


# In[64]:


df['A'].value_counts()


# One issue with categories is that, if a particular level of a category is not seen before, it can create an error. So you can pre-specify the categories you expect

# In[65]:


df_cat['B'] = pd.Categorical(list('aabb'), categories = ['a','b','c','d'])
df_cat['B'].value_counts()


# #### Re-organizing categories
# 
# In categorical data, there is often the concept of a "first" or "reference" category, and an ordering of categories. This tends to be important in both visualization as well as in regression modeling. Both aspects of a category can be addressed using the `reorder_categories` function.
# 
# In our earlier example, we can see that the `A` variable has 4 categories, with the "first" category being "a".

# In[66]:


df_cat.A


# Suppose we want to change this ordering to the reverse ordering, where
# "d" is the "first" category, and then it goes in reverse order. 

# In[67]:


df_cat['A'] = df_cat.A.cat.reorder_categories(['d','c','b','a'])
df_cat.A


# ### Missing data
# 
# Both `numpy` and `pandas` allow for missing values, which are a reality in data science. The missing values are coded as `np.nan`. Let's create some data and force some missing values

# In[68]:


df = pd.DataFrame(np.random.randn(5, 3), index = ['a','c','e', 'f','g'], columns = ['one','two','three']) # pre-specify index and column names
df['four'] = 20 # add a column named "four", which will all be 20
df['five'] = df['one'] > 0
df


# In[69]:


df2 = df.reindex(['a','b','c','d','e','f','g'])
df2.style.applymap(lambda x: 'background-color:yellow', subset = pd.IndexSlice[['b','d'],:])


# The code above is creating new blank rows based on the new index values, some of which are present in the existing data and some of which are missing.
# 
# We can create *masks* of the data indicating where missing values reside in a data set.

# In[70]:


df2.isna()


# In[71]:


df2['one'].notna()


# We can obtain complete data by dropping any row that has any missing value. This is called *complete case analysis*, and you should be very careful using it. It is *only* valid if we belive that the missingness is missing at random, and not related to some characteristic of the data or the data gathering process. 

# In[72]:


df2.dropna(how='any')


# You can also fill in, or *impute*, missing values. This can be done using a single value..

# In[73]:


out1 = df2.fillna(value = 5)

out1.style.applymap(lambda x: 'background-color:yellow', subset = pd.IndexSlice[['b','d'],:])


# or a computed value like a column mean

# In[74]:


df3 = df2.copy()
df3 = df3.select_dtypes(exclude=[object])   # remove non-numeric columns
out2 = df3.fillna(df3.mean())  # df3.mean() computes column-wise means

out2.style.applymap(lambda x: 'background-color:yellow', subset = pd.IndexSlice[['b','d'],:])


# You can also impute based on the principle of *last value carried forward* which is common in time series. This means that the missing value is imputed with the previous recorded value. 

# In[75]:


out3 = df2.fillna(method = 'ffill') # Fill forward

out3.style.applymap(lambda x: 'background-color:yellow', subset = pd.IndexSlice[['b','d'],:])


# In[76]:


out4 = df2.fillna(method = 'bfill') # Fill backward

out4.style.applymap(lambda x: 'background-color:yellow', subset = pd.IndexSlice[['b','d'],:])


# ## Data transformation
# 
# ### Arithmetic operations
# 
# If you have a `Series` or `DataFrame` that is all numeric, you can add or multiply single numbers to all the elements together.

# In[77]:


A = pd.DataFrame(np.random.randn(4,5))
print(A)


# In[78]:


print(A + 6)


# In[79]:


print(A * -10)


# If you have two compatible (same dimension) numeric `DataFrame`s, you can add, subtract, multiply and divide elementwise

# In[80]:


B = pd.DataFrame(np.random.randn(4,5) + 4)
print(A + B)


# In[81]:


print(A * B)


# If you have a `Series` with the same number of elements as the number of columns of a `DataFrame`, you can do arithmetic operations, with each element of the `Series` acting upon each column of the `DataFrame`

# In[82]:


c = pd.Series([1,2,3,4,5])
print(A + c)


# In[83]:


print(A * c)


# This idea can be used to standardize a dataset, i.e. make each column have mean 0 and standard deviation 1.

# In[84]:


means = A.mean(axis=0)
stds = A.std(axis = 0)

(A - means)/stds


# ### Concatenation of data sets
# 
# Let's create some example data sets

# In[85]:


df1 = pd.DataFrame({'A': ['a'+str(i) for i in range(4)],
    'B': ['b'+str(i) for i in range(4)],
    'C': ['c'+str(i) for i in range(4)],
    'D': ['d'+str(i) for i in range(4)]})

df2 =  pd.DataFrame({'A': ['a'+str(i) for i in range(4,8)],
    'B': ['b'+str(i) for i in range(4,8)],
    'C': ['c'+str(i) for i in range(4,8)],
    'D': ['d'+str(i) for i in range(4,8)]})
df3 =  pd.DataFrame({'A': ['a'+str(i) for i in range(8,12)],
    'B': ['b'+str(i) for i in range(8,12)],
    'C': ['c'+str(i) for i in range(8,12)],
    'D': ['d'+str(i) for i in range(8,12)]})


# We can concatenate these `DataFrame` objects by row

# In[86]:


row_concatenate = pd.concat([df1, df2, df3])
print(row_concatenate)


# This stacks the dataframes together. They are literally stacked, as is evidenced by the index values being repeated. 

# This same exercise can be done by the `append` function

# In[87]:


df1.append(df2).append(df3)


# Suppose we want to append a new row to `df1`. Lets create a new row.

# In[88]:


new_row = pd.Series(['n1','n2','n3','n4'])
pd.concat([df1, new_row])


# That's a lot of missing values. The issue is that the we don't have column names in the `new_row`, and the indices are the same, so pandas tries to append it my making a new column. The solution is to make it a `DataFrame`.

# In[89]:


new_row = pd.DataFrame([['n1','n2','n3','n4']], columns = ['A','B','C','D'])
print(new_row)


# In[90]:


pd.concat([df1, new_row])


# or

# In[91]:


df1.append(new_row)


# #### Adding columns

# In[92]:


pd.concat([df1,df2,df3], axis = 1)


# The option `axis=1` ensures that concatenation happens by columns. The default value `axis = 0` concatenates by rows.

# Let's play a little game. Let's change the column names of `df2` and `df3` so they are not the same as `df1`.

# In[93]:


df2.columns = ['E','F','G','H']
df3.columns = ['A','D','F','H']
pd.concat([df1,df2,df3])


# Now pandas ensures that all column names are represented in the new data frame, but with missing values where the row indices and column indices are mismatched. Some of this can be avoided by only joining on common columns. This is done using the `join` option ir `concat`. The default value is 'outer`, which is what you see. above

# In[94]:


pd.concat([df1, df3], join = 'inner')


# You can do the same thing when joining by rows, using `axis = 0` and `join="inner"` to only join on rows with matching indices. Reminder that the indices are just labels and happen to be the row numbers by default. 

# ### Merging data sets

# For this section we'll use a set of data from a survey, also used by Daniel Chen in "Pandas for Everyone"

# In[95]:


person = pd.read_csv('data/survey_person.csv')
site = pd.read_csv('data/survey_site.csv')
survey = pd.read_csv('data/survey_survey.csv')
visited = pd.read_csv('data/survey_visited.csv')


# In[96]:


print(person)


# In[97]:


print(site)


# In[98]:


print(survey)


# In[99]:


print(visited)


# There are basically four kinds of joins:
# 
# | pandas | R          | SQL         | Description                     |
# | ------ | ---------- | ----------- | ------------------------------- |
# | left   | left_join  | left outer  | keep all rows on left           |
# | right  | right_join | right outer | keep all rows on right          |
# | outer  | outer_join | full outer  | keep all rows from both         |
# | inner  | inner_join | inner       | keep only rows with common keys |

# ![](graphs/joins.png)
# 
# The terms `left` and `right` refer to which data set you call first and second respectively. 
# 
# We start with an left join
# 

# In[100]:


s2v_merge = survey.merge(visited, left_on = 'taken',right_on = 'ident', how = 'left')


# In[101]:


print(s2v_merge)


# Here, the left dataset is `survey` and the right one is `visited`. Since we're doing a left join, we keed all the rows from `survey` and add columns from `visited`, matching on the common key, called "taken" in one dataset and "ident" in the other. Note that the rows of `visited` are repeated as needed to line up with all the rows with common "taken" values. 
# 
# We can now add location information, where the common key is the site code

# In[102]:


s2v2loc_merge = s2v_merge.merge(site, how = 'left', left_on = 'site', right_on = 'name')
print(s2v2loc_merge)


# Lastly, we add the person information to this dataset.

# In[103]:


merged = s2v2loc_merge.merge(person, how = 'left', left_on = 'person', right_on = 'ident')
print(merged.head())


# You can merge based on multiple columns as long as they match up. 

# In[104]:


ps = person.merge(survey, left_on = 'ident', right_on = 'person')
vs = visited.merge(survey, left_on = 'ident', right_on = 'taken')
print(ps)


# In[105]:


print(vs)


# In[106]:


ps_vs = ps.merge(vs, 
                left_on = ['ident','taken', 'quant','reading'],
                right_on = ['person','ident','quant','reading']) # The keys need to correspond
ps_vs.head()


# Note that since there are common column names, the merge appends `_x` and `_y` to denote which column came from the left and right, respectively.
# 

# ### Tidy data principles and reshaping datasets
# 
# The tidy data principle is a principle espoused by Dr. Hadley Wickham, one of the foremost R developers. [Tidy data](http://vita.had.co.nz/papers/tidy-data.pdf) is a structure for datasets to make them more easily analyzed on computers. The basic principles are
# 
# + Each row is an observation
# + Each column is a variable
# + Each type of observational unit forms a table
# 
# > Tidy data is tidy in one way. Untidy data can be untidy in many ways
# 
# Let's look at some examples.

# In[107]:


from glob import glob
filenames = sorted(glob('data/table*.csv')) # find files matching pattern. I know there are 6 of them
table1, table2, table3, table4a, table4b, table5 = [pd.read_csv(f) for f in filenames] # Use a list comprehension


# This code imports data from 6 files matching a pattern. Python allows multiple assignments on the left of the `=`, and as each dataset is imported, it gets assigned in order to the variables on the left. In the second line I sort the file names so that they match the order in which I'm storing them in the 3rd line. The function `glob` does pattern-matching of file names. 
# 
# The following tables refer to the number of TB cases and population in Afghanistan, Brazil and China in 1999 and 2000

# In[108]:


print(table1)


# In[109]:


print(table2)


# In[110]:


print(table3)


# In[111]:


print(table4a) # cases


# In[112]:


print(table4b) # population


# In[113]:


print(table5)


# **Exercise:** Describe why and why not each of these datasets are tidy.

# ### Melting (unpivoting) data
# 
# Melting is the operation of collapsing multiple columns into 2 columns, where one column is formed by the old column names, and the other by the corresponding values. Some columns may be kept fixed and their data are repeated to maintain the interrelationships between the variables.
# 
# We'll start with loading some data on income and religion in the US from the Pew Research Center.

# In[114]:


pew = pd.read_csv('data/pew.csv')
print(pew.head())


# This dataset is considered in "wide" format. There are several issues with it, including the fact that column headers have data. Those column headers are income groups, that should be a column by tidy principles. Our job is to turn this dataset into "long" format with a column for income group. 
# 
# We will use the function `melt` to achieve this. This takes a few parameters:
# 
# + **id_vars** is a list of variables that will remain as is
# + **value_vars** is a list of column nmaes that we will melt (or unpivot). By default, it will melt all columns not mentioned in id_vars
# + **var_name** is a string giving the name of the new column created by the headers (default: `variable`)
# + **value_name** is a string giving the name of the new column created by the values (default: `value`)
# 

# In[115]:


pew_long = pew.melt(id_vars = ['religion'], var_name = 'income_group', value_name = 'count')
print(pew_long.head())


# ### Separating columns containing multiple variables
# 
# We will use an Ebola dataset to illustrate this principle

# In[116]:


ebola = pd.read_csv('data/country_timeseries.csv')
print(ebola.head())


# Note that for each country we have two columns -- one for cases (number infected) and one for deaths. Ideally we want one column for country, one for cases and one for deaths. 
# 
# The first step will be to melt this data sets so that the column headers in question from a column and the corresponding data forms a second column.

# In[117]:


ebola_long = ebola.melt(id_vars = ['Date','Day'])
print(ebola_long.head())


# We now need to split the data in the `variable` column to make two columns. One will contain the country name and the other either Cases or Deaths. We will use some string manipulation functions that we will see later to achieve this.

# In[118]:


variable_split = ebola_long['variable'].str.split('_', expand=True) # split on the `_` character
print(variable_split[:5])


# The `expand=True` option forces the creation of an `DataFrame` rather than a list

# In[119]:


type(variable_split)


# We can now concatenate this to the original data

# In[120]:


variable_split.columns = ['status','country']

ebola_parsed = pd.concat([ebola_long, variable_split], axis = 1)

ebola_parsed.drop('variable', axis = 1, inplace=True) # Remove the column named "variable" and replace the old data with the new one in the same location

print(ebola_parsed.head())


# ### Pivot/spread datasets
# 
# If we wanted to, we could also make two columns based on cases and deaths, so for each country and date you could easily read off the cases and deaths. This is achieved using the `pivot_table` function.
# 
# In the `pivot_table` syntax, `index` refers to the columns we don't want to change, `columns` refers to the column whose values will form the column names of the new columns, and `values` is the name of the column that will form the values in the pivoted dataset. 

# In[121]:


ebola_parsed.pivot_table(index = ['Date','Day', 'country'], columns = 'status', values = 'value')


# This creates something called `MultiIndex` in the `pandas` `DataFrame`. This is useful in some advanced cases, but here, we just want a normal `DataFrame` back. We can achieve that by using the `reset_index` function.

# In[122]:


ebola_parsed.pivot_table(index = ['Date','Day','country'], columns = 'status', values = 'value').reset_index()


# Pivoting is a 2-column to many-column operation, with the number of columns formed depending on the number of unique values present in the column of the original data that is entered into the `columns` argument of `pivot_table`

# **Exercise:** Load the file `weather.csv` into Python and work on making it a tidy dataset. It requires melting and pivoting. The dataset comprises of the maximun and minimum temperatures recorded each day in 2010. There are lots of missing value. Ultimately we want columns for days of the month, maximum temperature and minimum tempearture along with the location ID, the year and the month.

# ## Data aggregation and split-apply-combine
# 
# We'll use the Gapminder dataset for this section

# In[123]:


df = pd.read_csv('data/gapminder.tsv', sep = '\t') # data is tab-separated, so we use `\t` to specify that


# The paradigm we will be exploring is often called *split-apply-combine* or MapReduce or grouped aggregation. The basic idea is that you split a data set up by some feature, apply a recipe to each piece, compute the result, and then put the results back together into a dataset. This can be described in teh following schematic.

# ![](graphs/split-apply-combine.png)

# `pandas` is set up for this. It features the `groupby` function that allows the "split" part of the operation. We can then apply a function to each part and put it back together. Let's see how.

# In[124]:


df.head()


# In[125]:


f"This dataset has {len(df['country'].unique())} countries in it"


# One of the variables in this dataset is life expectancy at birth, `lifeExp`. Suppose we want to find the average life expectancy of each country over the period of study.

# In[126]:


df.groupby('country')['lifeExp'].mean()


# So what's going on here? First, we use the `groupby` function, telling `pandas` to split the dataset up by values of the column `country`.

# In[127]:


df.groupby('country')


# `pandas` won't show you the actual data, but will tell you that it is a grouped dataframe object. This means that each element of this object is a `DataFrame` with data from one country.

# In[128]:


df.groupby('country').ngroups


# In[129]:


df.groupby('country').get_group('United Kingdom')


# In[130]:


type(df.groupby('country').get_group('United Kingdom'))


# In[131]:


avg_lifeexp_country = df.groupby('country').lifeExp.mean()
avg_lifeexp_country['United Kingdom']


# In[132]:


df.groupby('country').get_group('United Kingdom').lifeExp.mean()


# Let's look at if life expectancy has gone up over time, by continent

# In[133]:


df.groupby(['continent','year']).lifeExp.mean()


# In[134]:


avg_lifeexp_continent_yr = df.groupby(['continent','year']).lifeExp.mean().reset_index()
avg_lifeexp_continent_yr


# In[135]:


type(avg_lifeexp_continent_yr)


# The aggregation function, in this case `mean`, does both the "apply" and "combine" parts of the process.

# We can do quick aggregations with `pandas`

# In[136]:


df.groupby('continent').lifeExp.describe()


# In[137]:


df.groupby('continent').nth(10) # Tenth observation in each group


# You can also use functions from other modules, or your own functions in this aggregation work.

# In[138]:


df.groupby('continent').lifeExp.agg(np.mean)


# In[139]:


def my_mean(values):
    n = len(values)
    sum = 0
    for value in values:
        sum += value
    return(sum/n)

df.groupby('continent').lifeExp.agg(my_mean)


# You can do many functions at once

# In[140]:


df.groupby('year').lifeExp.agg([np.count_nonzero, np.mean, np.std])


# You can also aggregate on different columns at the same time by passing a `dict` to the `agg` function

# In[141]:


df.groupby('year').agg({'lifeExp': np.mean,'pop': np.median,'gdpPercap': np.median}).reset_index()


# #### Transformation

# You can do grouped transformations using this same method. We will compute the z-score for each year, i.e. we will substract the average life expectancy and divide by the standard deviation

# In[142]:


def my_zscore(values):
    m = np.mean(values)
    s = np.std(values)
    return((values - m)/s)


# In[143]:


df.groupby('year').lifeExp.transform(my_zscore)


# In[144]:


df['lifeExp_z'] = df.groupby('year').lifeExp.transform(my_zscore)


# In[145]:


df.groupby('year').lifeExp_z.mean()


# #### Filter

# We can split the dataset by values of one variable, and filter out those splits that fail some criterion. The following code only keeps countries with a population of at least 10 million at some point during the study period

# In[146]:


df.groupby('country').filter(lambda d: d['pop'].max() > 10000000)


# 
# 
# 
# 
# 
# 
# 
