from pylab import *
import Image
import pyopencl as cl
import sys
import os.path

if not len(sys.argv) > 1:
    exit('Rage quit!')

target_file = sys.argv[1]

original = array(Image.open(target_file).convert('RGBA'))
height, width, channels = original.shape
original.dtype = np.uint32
buffer = original.flatten()

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
d_orig = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=buffer)
d_grey = cl.Buffer(ctx, mf.READ_WRITE, width*height)
d_mask = cl.Buffer(ctx, mf.READ_WRITE, width*height)
d_masked = cl.Buffer(ctx, mf.WRITE_ONLY, original.nbytes)

prg = cl.Program(ctx, """
    __kernel void grayscale(__global const uint* orig,
        __global uchar* dest,
        uint const width, uint const height)
    {
        int pos = get_global_id(0);
        int y = pos / width;
        int x = pos - (y*width);
        uchar* colors = (uchar *) &(orig[pos]);
        uchar r = colors[0];
        uchar g = colors[1];
        uchar b = colors[2];
        dest[pos] = .299f * r + .587f * g + .114f * b;
    }

    __kernel void threshold(__global const uchar* orig,
        __global bool* dest, uint const width,
        uint const height, uchar const threshold)
    {
        int pos = get_global_id(0);
        dest[pos] = (threshold <= orig[pos]);
    }

    __kernel void mask(__global const uint* orig,
        __global const bool* mask, __global uint* dest,
        uint const width, uint const height)
    {
        int pos = get_global_id(0);
        if (!(mask[pos])) {
            dest[pos] = 0;
        } else {
            dest[pos] = orig[pos];
        }
    }
    """).build()

prg.grayscale(
    queue, buffer.shape, None, d_orig, d_grey,
    np.uint32(width), np.uint32(height))

prg.threshold(
    queue, buffer.shape, None, d_grey, d_mask,
    np.uint32(width), np.uint32(height), np.uint8(225))

prg.mask(
    queue, buffer.shape, None, d_orig, d_mask, d_masked,
    np.uint32(width), np.uint32(height))

masked = np.empty((height, width), dtype=np.uint32)
cl.enqueue_copy(queue, masked, d_masked)

# Image read flips y-axis... flipping back now
# The copy is because we require the underlying data to be ordered right
# Image.frombuffer('RGBA', (width, height), result[::-1, :, :].copy()).save('../output/grey_butner.png')
# Image.frombuffer('L', (width, height), grey[::-1, :].copy()).save('../output/grey_%s' % os.path.basename(target_file))
Image.frombuffer('RGBA', (width, height), masked).save('../output/threshold_%s' % os.path.basename(target_file))
