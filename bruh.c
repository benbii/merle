#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

// Return a size that would hold the output if result size > outsz.
size_t base64_encode(const unsigned char *src, size_t len,
                     unsigned char *out, size_t outsz) {
  static const unsigned char base64_table[65] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const size_t olen = 4 * ((len + 2) / 3); /* 3-byte blocks to 4-byte */
  if (olen > outsz) return olen;
  // std::string outStr; outStr.resize(olen);
  const unsigned char *end = src + len, *in = src;
  unsigned char *pos = out;

  while (end - in >= 3) {
    *pos++ = base64_table[in[0] >> 2];
    *pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
    *pos++ = base64_table[((in[1] & 0x0f) << 2) | (in[2] >> 6)];
    *pos++ = base64_table[in[2] & 0x3f];
    in += 3;
  }

  if (end - in) {
    *pos++ = base64_table[in[0] >> 2];
    if (end - in == 1) {
      *pos++ = base64_table[(in[0] & 0x03) << 4];
      *pos++ = '=';
    } else {
      *pos++ = base64_table[((in[0] & 0x03) << 4) | (in[1] >> 4)];
      *pos++ = base64_table[(in[1] & 0x0f) << 2];
    }
    *pos++ = '=';
  }
  return olen;
}

// Return a size that would hold the output if result size > outsz.
size_t b64decode(const void *data, const size_t len, char *out, size_t outsz) {
  static const int B64index[256] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  62, 63, 62, 62, 63, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,
    7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 0,  0,  0,  0,  63, 0,  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51};

  unsigned char *p = (unsigned char *)data;
  int pad = len > 0 && (len % 4 || p[len - 1] == '=');
  const size_t L = ((len + 3) / 4 - pad) * 4;
  size_t realsz = L / 4 * 3 + pad;
  // std::string str(L / 4 * 3 + pad, '\0');
  if (realsz > outsz)
    return realsz + 4; // must be sufficient

  for (size_t i = 0, j = 0; i < L; i += 4) {
    int n = B64index[p[i]] << 18 | B64index[p[i + 1]] << 12 |
            B64index[p[i + 2]] << 6 | B64index[p[i + 3]];
    out[j++] = n >> 16;
    out[j++] = n >> 8 & 0xFF;
    out[j++] = n & 0xFF;
  }
  if (pad) {
    int n = B64index[p[L]] << 18 | B64index[p[L + 1]] << 12;
    out[realsz - 1] = n >> 16;
    if (len > L + 2 && p[L + 2] != '=') {
      n |= B64index[p[L + 2]] << 6;
      out[realsz++] = n >> 8 & 0xFF;
      if (realsz > outsz)
        return realsz + 4;
    }
  }
  return realsz;
}

int main(int argc, char **argv) {
  const uint8_t *in = ;// mmap all file content of argv[1]
  size_t outSz = base64_encode(in, inSz, NULL, 0);
  uint8_t *out = malloc(outSz);
  (void)base64_encode(in, inSz, out, outSz);
  munmap(in, inSz);

  size_t nRow = atol(argv[2]), nCol = atol(argv[3]), usl = atol(argv[4]);
}
