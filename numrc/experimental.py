import numpy as np

def distort_entry(entry):
    r = lambda x, y: np.random.randint(x, y + 1)

    # Element-wise
    entry = entry.rotate(np.radians(r(-25, 25)))
    entry = entry.squeeze(r(8, 10) / 10.0, r(8, 10) / 10.0)
    entry = entry.corner(r(-1, 1), r(-1, 1))

    # Post processing
    entry = entry.noise(r(1, 10) / 10.0 * 0.23)
    if r(1, 10) > 5:
        entry = entry.invert()

    return entry

def iterate_view(m, db):
    for entry in db:
        entry.print()
        print("Answer:", m.recognize(entry))
        input("...")
