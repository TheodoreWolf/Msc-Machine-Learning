def fibonacci(n):
    a = 0
    b = 1
    series = []
    for x in range(0, n):
        series.append(a)
        c = a + b
        a = b
        b = c

    print(series)

fibonacci(50)
f = fibonacci

f(50)