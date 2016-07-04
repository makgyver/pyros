# pyros

### How to build and evaluate a recommender

First of all you have to load the dataset. This module provides useful methods
for reading Comma Separated Values (CSV) files.

```python
reader = CSVReader("path\to\the\csv\file", " ")
data = ds.UDataset(Mapping(), Mapping())
reader.read(dataset, True) #True means that the ratings are binary
```

the code above reads the content of the given CSV file (space separated) and
saves it in the dataset variable. In the example the dataset is user-centered,
that is ratings are stored as set of items rated by a user.

Once the dataset is ready the recommender can be instanciated.
Currently the module offers, beyond the common baselines (e.g., popularity-based),
the following recommendation algorithms:

* Matrix-based implementation of the algorithm described in 
"Efficient Top-N Recommendation for Very Large Scale Binary Rated Datasets"
by F. Aiolli, which is based on the asymmetric cosine similarity

```python
rec = exp.I2I_Asym_Cos(data, alpha, q)
```

where 'alpha' is the asimmetric weight and 'q' the locality parameter.

* "Convex AUC Optimization for Top-N Recommendation with Implicit Feedback"
by F. Aiolli

```python
rec = exp.CF_OMD(data, lambda_p, lambda_n, sparse)
```

where 'lambda_p', 'lambda_n' are respectively the regularization terms for the
positives and negatives distribution, while 'sparse' is a boolean parameter that
says whether to use a sparse matrix implementation or not.

* Implementation of the algorithm (which is a simplification of CF-OMD) described in
"Kernel based collaborative filtering for very large scale top-N item recommendation"
by M.Polato and F. Aiolli

```python
exp.ECF_OMD(data, lambda_p, sparse)
```

where the parameters has the same meaning as in (2) but in this one 'lambda_n'
is not required (it is assumed to be +inf).

* Implementation of the algorithm (which is a "kernelification" of ECF-OMD) described in
"Kernel based collaborative filtering for very large scale top-N item recommendation"
by M.Polato and F. Aiolli

```python
import utils as ut
K = ut.kernels.normalize(ut.kernels.linear(data.to_cvxopt_matrix()))
rec = exp.CF_KOMD(data, K, lambda_p, sparse)
```

in this case a kernel 'K' is required as parameter. The code shows an example of linear kernel
built using the support methods provided by the 'utils' module.


After the instanciation of the recommender it has to be trained:

```python
rec.train(users)
```

where 'users' is the list of users for which the items ranking will
be calculated.

Finally, the evaluation step is:

```python
import core.evaluation as ev
result = ev.evaluate(rec, data_test)
```

where 'data_test' is the test dataset which contains the ratings
to predict (unknown at training time!!).
The evaluation is done using AUC, mAP and NDCG.

For more details please refer to the papers and to the code.


### For the lazy people :)

In order to try the recommendation algorithm with the default
dataset use the following command

```sh
$ python main.py ./datasets/ml1m.tr ./datasets/ml1m.te
```

### Version
0.9.0

### Tech

Pyros requires the following python modules:

* [Scikit-learn]
* [Numpy]
* [CVXOPT]

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Scikit-learn]: <http://scikit-learn.org/stable/>
   [Numpy]: <http://www.numpy.org/>
   [CVXOPT]: <http://cvxopt.org/>

