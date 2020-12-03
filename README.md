# NumberRecognition
Experimental project about combining neural network with genetic algorithm

```python
>>> import numrc.mlp as mlp
>>> import numrc.mnist as mn
>>> import numrc.experimental as exp

>>> db = mn.Database.load("image", "label")
>>> db[0].print()
>>> m1 = mlp.SigmoidMLP(30)
>>> m1.recognize(db[0])
>>> exp.test_distortion(db)
```

## OpenCL

OpenCL is planned to be used for image deformation.

## Results

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
