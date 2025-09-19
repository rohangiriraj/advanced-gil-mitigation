# image_cython.pyx
from cython.parallel import prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def grayscale_cython(unsigned char[:, :, ::1] source_image, unsigned char[:, ::1] dest_image):
    """
    Performs grayscale conversion using Cython and OpenMP.
    The source_image is the input (color), and the dest_image is the output (grayscale).
    """
    cdef int height = source_image.shape[0]
    cdef int width = source_image.shape[1]
    cdef int x, y

    # Release the GIL to allow for true parallel processing
    with nogil:
        # The prange function parallelizes the outer loop across multiple CPU cores
        for y in prange(height, schedule='static'):
            for x in range(width):
                # Direct C-level access to pixel data
                dest_image[y, x] = int(
                    source_image[y, x, 0] * 0.299 +  # Red
                    source_image[y, x, 1] * 0.587 +  # Green
                    source_image[y, x, 2] * 0.114    # Blue
                )
