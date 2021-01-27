# NumberRecognition
Experimental project about combining neural network with genetic algorithm

```python
>>> import numrc.mlp as mlp
>>> import numrc.mnist as mn
>>> import numrc.experimental as exp

# Print image using ANSI escape codes
>>> db = mn.Database.load("image", "label")
>>> db[0].print()

>>> m1 = mlp.SigmoidMLP(30)
>>> m1.recognize(db[0])

>>> m1.train(db, 50, 500, 6.5)
>>> m1.recognize(db[0])
```

## Genetic Algorithm

```python
>>> offs = a1.evolve(a1, 20, 15, exp.distort_db(db))

Epoch 0 done:
- [95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4,
95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4, 95.4,
95.4, 95.4, 95.4, 95.4, 95.4, 95.2]
...
```

Reproduction model is defined as follows:

* Chromosome is a set of every single weight and bias parameter
* The best four are selected as the two pairs of parents for next generation

As for crossover and mutation, each parameter is determined in the following manner:

* 50% chance to be derived from parent B
* Chance of *1 in entire chromosome length* to be mutated.\* New value is randomly selected from N(mu, sigma^2) for each layer's weight and bias group.\*\*
* Rest are derived from parent A

<sub>\* Mutation is done in order to prevent being stuck at local minima.<br>
\*\* It is an arbitrarily chosen method for mutating</sub>


## OpenCL

OpenCL is primarily used for image deformation in order to augment the database. OpenCL was adopted to speed up the deformation process and thus provide the MLP with new data more frequently.

```python
>>> import numrc.experimental as exp
>>> import numrc.mnist as mn

>>> db = mn.Database.load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte")
>>> db[0].print()

# Randomly deform every image
>>> db2 = exp.distort_db(db)
>>> db2[0].print()

# Uniformly deform all the 60,000 images
>>> db3 = db.clone()
>>> db3.start_filters()
>>> db3.rotate(3.14 / 4) # rotate 45 degrees clockwise
>>> db3.scale(0.8, 0.8)
>>> db3.corner(1, 1) # corner the content to the top right(+x, +y)
>>> db3.noise(0.2) # add random values to pixels (-0.2 to 0.2)
>>> db3.invert(True)
>>> db3.flush_filters()
>>> db3[0].print()
```

#### Benchmarks

Randomly rotating, scaling, cornering, applying noise, inverting every image in the standarad database of 60,000 images:

Python

* Synchronized sequential CPU operations
* Big Sur i5-5257U CPU @ 2.70GHz
* ~4 minutes to process images

OpenCL

* Parallel GPU operations
* Big Sur Intel(R) Iris(TM) Graphics 6100
* 5 seconds


## Results

### `ga-92_56.tar.gz`

```python
>>> offs = m1.evolve(m2, 5, 100, tdb)
```

```markdown
Epoch 0 done:
- 1st score: 92.29
- 2nd score: 92.26
- 3rd score: 92.21
- 4th score: 92.2
Epoch 1 done:
- 1st score: 92.35
- 2nd score: 92.32
- 3rd score: 92.31
- 4th score: 92.31
Epoch 2 done:
- 1st score: 92.44
- 2nd score: 92.43
- 3rd score: 92.43
- 4th score: 92.42
Epoch 3 done:
- 1st score: 92.44
- 2nd score: 92.44
- 3rd score: 92.44
- 4th score: 92.44
Epoch 4 done:
- 1st score: 92.56
- 2nd score: 92.49
- 3rd score: 92.49
- 4th score: 92.49
```

Methods used:

* Genetic algorithm with 92.23% and 91.62% that learned from augmented database

Insights learned:

* Offsprings showed better results than could their parents ever do with augmented database

Test result:

* 92.56%

### `aug-92_23.tar.gz`

Methods used:

* Image deformation in the following order
    * Rotate -35 to 35 degrees
    * Add noise -0.3 to 0.3
    * 50% chance of inverting the content
* Backpropagation

Test result against MNIST test database:

* 92.23%

### Deleted 01

Methods used:

* Image deformation in the following order
    * Rotate -25 to 25 degrees
    * Shrinking the image down to 80% at most for both axis
    * 66% chance of cornering the number to either positive or negative direction separately for both axis
    * Add noise -0.23 to 0.23
    * 50% chance of inverting
* Backpropagation

Test result against MNIST test database:

* ~51%

## Tested Environments

### 1. MBP

* macOS Big Sur
* i5-5257U CPU @ 2.70GHz
* Iris(TM) Graphics 6100
* Python 3.9.0

```text
Package    Version
---------- --------
appdirs    1.4.4
decorator  4.4.2
numpy      1.19.4
pip        20.3
pyopencl   2020.3.1
pytools    2020.4.3
setuptools 49.2.1
six        1.15.0
```

### 2. Google Colab

* Nvidia K80
* Python 3.6.9

```text
Package    Version
---------- --------
appdirs    1.4.4         
decorator  4.4.2        
numpy      1.19.5           
pip        19.3.1     
pyopencl   2020.3.1   
pytools    2021.1     
setuptools 51.3.3    
six        1.15.0
```
