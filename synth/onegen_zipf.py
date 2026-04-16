import numpy as np
import sys
import struct

def generate_zipf_samples(n_samples, s):
    """Generate samples from Zipf distribution using numpy's built-in function"""
    # numpy.random.zipf uses parameter 'a' which corresponds to our skewness 's'
    # It returns 1-based values, so we subtract 1 to make it 0-based
    samples = np.random.zipf(s, n_samples) - 1
    return samples

def get_bit_width(maxval):
    """Determine appropriate bit width based on maxval"""
    if maxval <= 255:
        return 8
    elif maxval <= 65535:
        return 16
    elif maxval <= 4294967295:
        return 32
    else:
        return 64

def write_binary_output(values, bit_width, outpath):
    """Write values to binary file with bit width header"""
    with open(outpath, 'wb') as f:
        # Write bit width as first 4 bytes (little-endian uint32)
        f.write(struct.pack('<I', bit_width))

        # Write values based on bit width
        if bit_width == 8:
            values_bytes = values.astype(np.uint8).tobytes()
        elif bit_width == 16:
            values_bytes = values.astype(np.uint16).tobytes()
        elif bit_width == 32:
            values_bytes = values.astype(np.uint32).tobytes()
        elif bit_width == 64:
            values_bytes = values.astype(np.uint64).tobytes()

        f.write(values_bytes)

def get_dtype(bit_width):
    """Get numpy dtype for given bit width"""
    if bit_width == 8:
        return np.uint8
    elif bit_width == 16:
        return np.uint16
    elif bit_width == 32:
        return np.uint32
    else:
        return np.uint64

def main():
    if len(sys.argv) != 5:
        print("Usage: python scrapad.py <skewness> <maxval> <nrelem> <outpath>")
        sys.exit(1)

    try:
        skewness = float(sys.argv[1])
        maxval = int(sys.argv[2])
        nrelem = int(sys.argv[3])
        outpath = sys.argv[4]
    except ValueError:
        print("Error: Invalid arguments. Expected: <float> <int> <int> <string>")
        sys.exit(1)

    # Determine bit width
    bit_width = get_bit_width(maxval)
    dtype = get_dtype(bit_width)

    # Write in chunks to reduce memory usage
    chunk_size = 6_000_000

    with open(outpath, 'wb') as f:
        # Write bit width as first 4 bytes (little-endian uint32)
        f.write(struct.pack('<I', bit_width))

        # Generate and write in chunks
        remaining = nrelem
        while remaining > 0:
            current_chunk = min(chunk_size, remaining)
            samples = generate_zipf_samples(current_chunk, skewness)
            samples = samples % maxval
            f.write(samples.astype(dtype).tobytes())
            remaining -= current_chunk

if __name__ == "__main__":
    main()
