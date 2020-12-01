import numpy as np

def distort_entry(entry):
    r = lambda x, y: np.random.randint(x, y + 1)

    entry = entry.noise(r(1, 10) / 10.0 * 0.3)
    entry = entry.rotate(np.radians(r(-35, 35)))

    if r(1, 10) > 5:
        entry = entry.invert()

    return entry

def iterate_test(m, tdb):
    for entry in tdb:
        entry.print()
        print("Answer:", m.recognize(entry))
        input("...")
