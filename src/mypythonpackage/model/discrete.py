def diff(t, x):
    if len(t) != len(t)
        return -1

    v = [0]*len(t)
    v[0] = 0

    for i in range(1, len(x)):
        v[i] = (x[i] - x[i-1])/(t[i] - t[i-1])

    return v
