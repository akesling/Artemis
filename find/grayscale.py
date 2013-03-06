from pylab import *
import pyopencl as cl
import cairo

# Imread flips the y-axis, so we have to flip it back...
original = imread('../data/over_exposed_butner.jpg')[::-1, :, ::-1]
original = np.dstack(
    (original, 255*ones(original.shape[:2], dtype=np.uint8))
    )#.copy()
height, width, channels = original.shape
original.dtype = np.uint32
buffer = original.flatten()

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
d_orig = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=buffer)
d_dest = cl.Buffer(ctx, mf.WRITE_ONLY, buffer.nbytes)

prg = cl.Program(ctx, """
    __kernel void grayscale(__global const uint* orig,
        __global uint* dest,
        uint const width, uint const height)
    {
      int x = get_global_id(0);
      int y = get_global_id(1);
      int pos = x + y*width;
      dest[pos] = orig[pos];
    }
    """).build()

prg.grayscale(
    queue, buffer.shape, None, d_orig, d_dest,
    np.uint32(width), np.uint32(height))

result = np.empty_like(buffer)
cl.enqueue_copy(queue, result, d_dest)
result.dtype = np.uint8
result.shape = (height, width, channels)

surface = cairo.ImageSurface.create_for_data(
    result, cairo.FORMAT_ARGB32, width, height)
surface.write_to_png('../data/grey_butner.png')
