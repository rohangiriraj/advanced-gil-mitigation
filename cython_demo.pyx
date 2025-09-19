# cython_demo.pyx

# cdef is used to declare C variables
def sum_of_squares(int n):
    """
    Calculates the sum of squares up to n.
    This is much faster in Cython because the loop runs entirely in C.
    """
    cdef long long total = 0
    cdef int i
    for i in range(1, n + 1):
        total += i * i
    return total
