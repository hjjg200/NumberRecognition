import numpy as np
import random

from .mnist import Database

def load_tdb():
    return Database.load("data/t10k-images.idx3-ubyte", \
        "data/t10k-labels.idx1-ubyte")

def distort_entry(entry):
    r = lambda x, y: np.random.randint(x, y + 1)

    # Element-wise
    entry = entry.rotate(np.radians(r(-17, 17)))
    #entry = entry.squeeze(r(18, 20) / 20.0, r(18, 20) / 20.0)
    #entry = entry.corner(r(-1, 1), r(-1, 1))

    # Post processing
    #entry = entry.noise(r(1, 10) / 10.0 * 0.15)
    #if r(1, 10) > 5:
        #entry = entry.invert()

    return entry

def iterate_over_db(db, f, rand=False):
    if rand:
        db = list(db)
        random.shuffle(db)
    for entry in db:
        f(entry)
        input("...")

def test_distortion(db):
    iterate_over_db(db, lambda e: distort_entry(e).print(), True)

def iterate_view(m, db, **kwargs):
    def f(e):
        nonlocal m
        e.print()
        print("Answer:", m.recognize(e))
    iterate_over_db(db, f, **kwargs)
