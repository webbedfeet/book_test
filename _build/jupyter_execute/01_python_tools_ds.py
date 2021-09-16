#!/usr/bin/env python
# coding: utf-8

# 
# # Python tools for data science
# (last updated 2020-05-18)
# 
# ## The PyData Stack
# 
# The Python Data Stack comprises a set of packages that makes Python a powerful data science language. These include
# 
# + Numpy: provides arrays and matrix algebra
# + Scipy: provides scientific computing capabilities
# + matplotlib: provides graphing capabilities
# 
# These were the original stack that was meant to replace Matlab. However, these were meant to tackle purely numerical data, and the kinds of heterogeneous data we regularly face needed more tools. These were added more recently.
# 
# + Pandas: provides data analytic structures like the data frame, as well as basic descriptive statistical capabilities
# + statsmodels: provides a fairly comprehensive set of statistical functions
# + scikit-learn: provides machine learning capabilities
# 
# This is the basic stack of packages we will be using in this workshop. Additionally we will use a few packages that add some functionality to the data science process. These include
# 
# + seaborn: Better statistical graphs
# 
# + plotly: Interactive graphics
# 
# + biopython: Python for bioinformatics
# 
# We may also introduce the package `rpy2` which allows one to run R from within Python. This can be useful since many bioinformatic pipelines are already implemented in R. 
# 
# > The [PyData stack](https://scipy.org) also includes `sympy`, a symbolic mathematics package emulating Maple
# 
# ## Numpy (numerical and scientific computing)
# 
# We start by importing the Numpy package into Python using the alias `np`. 

# In[1]:


import numpy as np


# Numpy provides both arrays (vectors, matrices, higher dimensional arrays) and vectorized functions which are very fast. Let's see how this works.

# In[2]:


z = [1,2,3,4,5,6,7,8,9.3,10.6] # This is a list
z_array = np.array(z)
z_array


# Now, we have already seen functions in Python earlier. In Numpy, there are functions that are optimized for arrays, that can be accessed directly from the array objects. This is an example of *object-oriented programming* in Python, where functions are provided for particular *classes* of objects, and which can be directly accessed from the objects. We will use several such functions over the course of this workshop, but we won't actually talk about how to do this program development here. 

# > Numpy functions are often very fast, and are *vectorized*, i.e., they are written to work on vectors of numbers rather than single numbers. This is an advantage in data science since we often want to do the same operation to all elements of a column of data, which is essentially a vector

# We apply the functions `sum`, `min` (minimum value) and `max` (maximum value) to `z_array`. 

# In[3]:


z_array.sum()


# In[4]:


z_array.min()


# In[5]:


z_array.max()


# The versions of these functions in Numpy are optimized for arrays and are quite a bit faster than the corresponding functions available in base Python. When doing data work, these are the preferred functions. 
# 
# These functions can also be used in the usual function manner:

# In[6]:


np.max(z_array)


# Calling `np.max` ensures that we are using the `max` function from numpy, and not the one in base Python. 
# 
# ### Numpy data types
# 
# Numpy arrays are homogeneous in type.

# In[7]:


np.array(['a','b','c'])


# In[8]:


np.array([1,2,3,6,8,29])


# But, what if we provide a heterogeneous list?

# In[9]:


y = [1,3,'a']
np.array(y)


# So what's going on here? Upon conversion from a heterogeneous list, numpy converted the numbers into strings. This is necessary since, by definition, numpy arrays can hold data of a single type. When one of the elements is a string, numpy casts all the other entities into strings as well. Think about what would happen if the opposite rule was used. The string 'a' doesn't have a corresponding number, while both numbers 1 and 3 have corresponding string representations, so going from string to numeric would create all sorts of problems. 

# > The advantage of numpy arrays is that the data is stored in a contiguous section of memory, and  you can be very efficient with homogeneous arrays in terms of manipulating them, applying functions, etc. However, `numpy` does provide a "catch-all" `dtype` called `object`, which can be any Python object. This `dtype` essentially is an array of pointers to actual data stored in different parts of the memory. You can get to the actual objects by extracting them. So one could do  <a name='object'></a>

# In[10]:


np.array([1,3,'a'], dtype='object')


# > which would basically be a valid `numpy` array, but would go back to the actual objects when used, much like a list. We can see this later if we want to transform a heterogeneous `pandas` `DataFrame` into a `numpy` array. It's not particularly useful as is, but it prevents errors from popping up during transformations from `pandas` to `numpy`.

# ### Generating data in numpy
# 
# We had seen earlier how we could generate a sequence of numbers in a list using `range`. In numpy, you can generate a sequence of numbers in an array using `arange` (which actually creates the array rather than provide an iterator like `range`).

# In[11]:


np.arange(10)


# You can also generate regularly spaced sequences of numbers between particular values

# In[12]:


np.linspace(start=0, stop=1, num=11) # or np.linspace(0, 1, 11)


# You can also do this with real numbers rather than integers.

# In[13]:


np.linspace(start = 0, stop = 2*np.pi, num = 10)


# More generally, you can transform lists into `numpy` arrays. We saw this above for vectors. For matrices, you can provide a list of lists. Note the double `[` in front and back. 

# In[14]:


np.array([[1,3,5,6],[4,3,9,7]])


# You can generate an array of 0's

# In[15]:


np.zeros(10)


# This can easily be extended to a two-dimensional array (a matrix), by specifying the dimension of the matrix as a tuple. 

# In[16]:


np.zeros((10,10))


# You can also generate a matrix of 1s in a similar manner.

# In[17]:


np.ones((3,4))


# In matrix algebra, the identity matrix is important. It is a square matrix with 1's on the diagonal and 0's everywhere else. 

# In[18]:


np.eye(4)


# You can also create numpy vectors directly from lists, as long as lists are made up of atomic elements of the same type. This means a list of numbers or a list of strings. The elements can't be more composite structures, generally. One exception is a list of lists, where all the lists contain the same type of atomic data, which, as we will see, can be used to create a matrix or 2-dimensional array.

# In[19]:


a = [1,2,3,4,5,6,7,8]
b = ['a','b','c','d','3']

np.array(a)


# In[20]:


np.array(b)


# #### Random numbers
# 
# Generating random numbers is quite useful in many areas of data science. All computers don't produce truly random numbers but generate *pseudo-random* sequences. These are completely deterministic sequences defined algorithmically that emulate the properties of random numbers. Since these are deterministic, we can set a *seed* or starting value for the sequence, so that we can exactly reproduce this sequence to help debug our code. To actually see how things behave in simulations we will often run several sequences of random numbers starting at different seed values. 
# 
# The seed is set by the `RandomState` function within the `random` submodule of numpy. Note that all Python names are case-sensitive. 

# In[21]:


rng = np.random.RandomState(35) # set seed
rng.randint(0, 10, (3,4))


# We have created a 3x4 matrix of random integers between 0 and 10 (in line with slicing rules, this includes 0 but not 10). 
# 
# We can also create a random sample of numbers between 0 and 1. 

# In[22]:


rng.random_sample((5,2))


# We'll see later how to generate random numbers from particular probability distributions. 
# 
# ### Vectors and matrices
# 
# Numpy generates arrays, which can be of arbitrary dimension. However the most useful are vectors (1-d arrays) and matrices (2-d arrays).
# 
# In these examples, we will generate samples from the Normal (Gaussian) distribution, with mean 0 and variance 1. 

# In[23]:


A = rng.normal(0,1,(4,5))


# We can compute some characteristics of this matrix's dimensions. The number of rows and columns are given by `shape`. 

# In[24]:


A.shape


# The total number of elements are given by `size`. 

# In[25]:


A.size


# If we want to create a matrix of 0's with the same dimensions as `A`, we don't actually have to compute its dimensions. We can use the `zeros_like` function to figure it out.

# In[26]:


np.zeros_like(A)


# We can also create vectors by only providing the number of rows to the random sampling function. The number of columns will be assumed to be 1. 

# In[27]:


B = rng.normal(0, 1, (4,))
B


# #### Extracting elements from arrays
# 
# The syntax for extracting elements from arrays is almost exactly the same as for lists, with the same rules for slices.
# 
# **Exercise:** State what elements of B are extracted by each of the following statements
# 
# ```
# B[:3]
# B[:-1]
# B[[0,2,4]]
# B[[0,2,5]]
# ```
# 
# For matrices, we have two dimensions, so you can slice by rows, or columns or both. 

# In[28]:


A


# We can extract the first column by specifying `:` (meaning everything) for the rows, and the index for the column (reminder, Python starts counting at 0)

# In[29]:


A[:,0]


# Similarly the 4th row can be extracted by putting the row index, and `:` for the column index. 

# In[30]:


A[3,:]


# All slicing operations work for rows and columns

# In[31]:


A[:2,:2]


# #### Array operations
# 
# We can do a variety of vector and matrix operations in `numpy`. 
# 
# First, all usual arithmetic operations work on arrays, like adding or multiplying an array with a scalar.

# In[32]:


A = rng.randn(3,5)
A


# In[33]:


A + 10


# We can also add and multiply arrays __element-wise__ as long as they are the same shape.

# In[34]:


B = rng.randint(0,10, (3,5))
B


# In[35]:


A + B


# In[36]:


A * B


# You can also do **matrix multiplication**. Recall what this is.
# 
# If you have a matrix $A_{m x n}$ and another matrix $B_{n x p}$, as long as the number of columns of $A$ and rows of $B$ are the same, you can multiply them ($C_{m x p} = A_{m x n}B_{n x p}$), with the (i,j)-th element of C being
# 
# $$ c_{ij} = \sum_{k=1}^n a_{ik}b_{kj}, i= 1, \dots, m; j = 1, \dots, p$$
# 
# In `numpy` the operant for matrix multiplication is `@`. 
# 
# In the above examples, `A` and `B` cannot be multiplied since they have incompatible dimensions. However, we can take the *transpose* of `B`, i.e. flip the rows and columns, to make it compatible with `A` for matrix multiplication.

# In[37]:


A @ np.transpose(B)


# In[38]:


np.transpose(A) @ B


# More generally, you can *reshape* a `numpy` array into a new shape, provided it is compatible with the number of elements in the original array.

# In[39]:


D = rng.randint(0,5, (4,4))
D


# In[40]:


D.reshape(8,2)


# In[41]:


D.reshape(1,16)


# This can also be used to cast a vector into a matrix. 

# In[42]:


e = np.arange(20)
E = e.reshape(5,4)
E


# > One thing to note in all the reshaping operations above is that the new array takes elements of the old array **by row**. See the examples above to convince yourself of that. 
# 

# #### Statistical operations on arrays
# 
# You can sum all the elements of a matrix using `sum`. You can also sum along rows or along columns by adding an argument to the `sum` function. 

# In[43]:


A = rng.normal(0, 1, (4,2))
A


# In[44]:


A.sum()


# You can sum along rows (i.e., down columns) with the option `axis = 0`

# In[45]:


A.sum(axis=0)


# You can sum along columns (i.e., across rows) with `axis = 1`.

# In[46]:


A.sum(axis=1)


# > Of course, you can use the usual function calls: `np.sum(A, axis = 1)`

# We can also find the minimum and maximum values.

# In[47]:


A.min(axis = 0)


# In[48]:


A.max(axis = 0)


# We can also find the **position** where the minimum and maximum values occur.

# In[49]:


A.argmin(axis=0)


# In[50]:


A.argmax(axis=0)


# We can sort arrays and also find the indices which will result in the sorted array. I'll demonstrate this for a vector, where it is more relevant

# In[51]:


a = rng.randint(0,10, 8)
a


# In[52]:


np.sort(a)


# In[53]:


np.argsort(a)


# In[54]:


a[np.argsort(a)]


# `np.argsort` can also help you find the 2nd smallest or 3rd largest value in an array, too.

# In[55]:


ind_2nd_smallest = np.argsort(a)[1]
a[ind_2nd_smallest]


# In[56]:


ind_3rd_largest = np.argsort(a)[-3]
a[ind_3rd_largest]


# You can also sort strings in this way.

# In[57]:


m = np.array(['Aram','Raymond','Elizabeth','Donald','Harold'])
np.sort(m)


# If you want to sort arrays **in place**, you can use the `sort` function in a different way.

# In[58]:


m.sort()
m


# #### Putting arrays together
# 
# We can put arrays together by row or column, provided the corresponding axes have compatible lengths. 

# In[59]:


A = rng.randint(0,5, (3,5))
B = rng.randint(0,5, (3,5))
print('A = ', A)
print('B = ', B)


# In[60]:


np.hstack((A,B))


# In[61]:


np.vstack((A,B))


# Note that both `hstack` and `vstack` take a **tuple** of arrays as input.

# #### Logical/Boolean operations
# 
# You can query a matrix to see which elements meet some criterion. In this example, we'll see which elements are negative.

# In[62]:


A < 0


# This is called **masking**, and is useful in many contexts. 
# 
# We can extract all the negative elements of A using

# In[63]:


A[A<0]


# This forms a 1-d array. You can also count the number of elements that meet the criterion

# In[64]:


np.sum(A<0)


# Since the entity `A<0` is a matrix as well, we can do row-wise and column-wise operations as well. 

# ### Beware of copies
# 
# One has to be a bit careful with copying objects in Python. By default, if you just assign one object to a new name, it does a *shallow copy*, which means that both names point to the same memory. So if you change something in the original, it also changes in the new copy. 

# In[65]:


A[0,:]


# In[66]:


A1 = A
A1[0,0] = 4
A[0,0]


# To actually create a copy that is not linked back to the original, you have to make a *deep copy*, which creates a new space in memory and a new pointer, and copies the original object to the new memory location

# In[67]:


A1 = A.copy()
A1[0,0] = 6
A[0,0]


# You can also replace sub-matrices of a matrix with new data, provided that the dimensions are compatible. (Make sure that the sub-matrix we are replacing below truly has 2 rows and 2 columns, which is what `np.eye(2)` will produce)

# In[68]:


A[:2,:2] = np.eye(2)
A


# #### Reducing matrix dimensions
# 
# Sometimes the output of some operation ends up being a matrix of one column or one row. We can reduce it to become a vector. There are two functions that can do that, `flatten` and `ravel`. 

# In[69]:


A = rng.randint(0,5, (5,1))
A


# In[70]:


A.flatten()


# In[71]:


A.ravel()


# So why two functions? I'm not sure, but they do different things behind the scenes. `flatten` creates a **copy**, i.e. a new array disconnected from `A`. `ravel` creates a **view**, so a representation of the original array. If you then changed a value after a `ravel` operation, you would also change it in the original array; if you did this after a `flatten` operation, you would not. 

# ### Broadcasting in Python

# Python deals with arrays in an interesting way, in terms of matching up dimensions of arrays for arithmetic operations. There are 3 rules:
# 
# 1. If two arrays differ in the number of dimensions, the shape of the smaller array is padded with 1s on its _left_ side
# 2. If the shape doesn't match in any dimension, the array with shape = 1 in that dimension is stretched to match the others' shape
# 3. If in any dimension the sizes disagree and none of the sizes are 1, then an error is generated

# In[72]:


A = rng.normal(0,1,(4,5))
B = rng.normal(0,1,5)


# In[73]:


A.shape


# In[74]:


B.shape


# In[75]:


A - B


# B is 1-d, A is 2-d, so B's shape is made into (1,5) (added to the left). Then it is repeated into 4 rows to make it's shape (4,5), then the operation is performed. This means that we subtract the first element of B from the first column of A, the second element of B from the second column of A, and so on.

# You can be explicit about adding dimensions for broadcasting by using `np.newaxis`.

# In[76]:


B[np.newaxis,:].shape


# In[77]:


B[:,np.newaxis].shape


# #### An example (optional, intermediate/advanced))
# 
# This can be very useful, since these operations are faster than for loops. For example:

# In[78]:


d = rng.random_sample((10,2))
d


# We want to find the Euclidean distance (the sum of squared differences) between the points defined by the rows. This should result in a 10x10 distance matrix

# In[79]:


d.shape


# In[80]:


d[np.newaxis,:,:]


# creates a 3-d array with the first dimension being of length 1

# In[81]:


d[np.newaxis,:,:].shape


# In[82]:


d[:, np.newaxis,:]


# creates a 3-d array with the 2nd dimension being of length 1

# In[83]:


d[:,np.newaxis,:].shape


# Now for the trick, using broadcasting of arrays. These two arrays are incompatible without broadcasting, but with broadcasting, the right things get repeated to make things compatible

# In[84]:


dist_sq = np.sum((d[:,np.newaxis,:] - d[np.newaxis,:,:]) ** 2)


# In[85]:


dist_sq.shape


# In[86]:


dist_sq


# Whoops! we wanted a 10x10 matrix, not a scalar. 

# In[87]:


(d[:,np.newaxis,:] - d[np.newaxis,:,:]).shape


# What we really want is the 10x10 distance matrix. 

# In[88]:


dist_sq = np.sum((d[:,np.newaxis,:] - d[np.newaxis,:,:]) ** 2, axis=2)


# You can verify what is happening by creating `D = d[:,np.newaxis,:]-d[np.newaxis,:,:]` and then looking at `D[:,:,0]` and `D[:,:,1]`. These are the difference between each combination in the first and second columns of d, respectively. Squaring and summing along the 3rd axis then gives the sum of squared differences. 

# In[89]:


dist_sq


# In[90]:


dist_sq.shape


# In[91]:


dist_sq.diagonal()


# ### Conclusions moving forward
# 
# It's important to understand numpy and arrays, since most data sets we encounter are rectangular. The notations and operations we saw in numpy will translate to data, except for the fact that data is typically heterogeneous, i.e., of different types. The problem with using numpy for modern data analysis is that if you have mixed data types, it will all be coerced to strings, and then you can't actually do any data analysis. 
# 
# The solution to this issue (which is also present in Matlab) came about with the `pandas` package, which is the main workhorse of data science in Python
# 
