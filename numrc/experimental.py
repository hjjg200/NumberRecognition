import numpy as np
import random

from .mnist import Database

def load_db():
    return Database.load("data/train-images.idx3-ubyte", \
        "data/train-labels.idx1-ubyte")

def load_tdb():
    return Database.load("data/t10k-images.idx3-ubyte", \
        "data/t10k-labels.idx1-ubyte")

def distort_db(db):

    db = db.clone()

    r = lambda x, y: np.random.randint(x, y + 1, len(db))

    db.start_filters()

    db.rotate(np.radians(r(-13, 13)))
    db.scale(r(18, 20) / 20.0, r(18, 20) / 20.0)
    #db.corner(r(-1, 1), r(-1, 1), 0.0)

    db.noise(r(5, 15) / 100.0)
    db.invert(r(0, 1))

    db.flush_filters()

    return db

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
