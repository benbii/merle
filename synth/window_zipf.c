#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Calculate window probabilities for Zipf distribution
 * @param skewness The skewness parameter for Zipf distribution
 * @param n The number of windows to calculate
 * @param winsize The size of each window
 * @return Array of probabilities for windows [x, x+winsize) where x in [0, n)
 */
double *window_zipf(double skewness, size_t n, size_t winsize) {
  // Allocate memory for the result array
  double *probabilities = (double *)malloc(n * sizeof(double));
  assert(probabilities);
  assert(skewness > 1.0);

  // OPTIMIZATION: Compute Dirichlet eta once (it's the same for all windows)
  // Inline dirichlet_eta computation
  double eta_sum = 0.0;
  double sign = 1.0;
  for (int k = 1; k <= 50000; k++) {
    eta_sum += sign / pow((double)k, skewness);
    sign *= -1.0; // Alternate signs
  }

  // OPTIMIZATION: Compute Riemann zeta once (same for all windows)
  // Inline riemann_zeta computation
  double denominator_factor = 1.0 - pow(2.0, 1.0 - skewness);
  double zeta_value = eta_sum / denominator_factor;

  // OPTIMIZATION: Precompute all k^(-s) values we'll need
  // Maximum k value we'll need is min(n + winsize, 100000)
  size_t max_k = n + winsize;
  if (max_k > 100000) max_k = 100000;

  double *k_powers = (double *)malloc((max_k + 1) * sizeof(double));
  assert(k_powers);

  // Precompute 1/k^s for all k values
  k_powers[0] = 0.0;  // Not used, but avoid issues
  for (size_t k = 1; k <= max_k; k++) {
    k_powers[k] = 1.0 / pow((double)k, skewness);
  }

  // Calculate probability for each window [x, x+winsize)
  for (size_t x = 0; x < n; x++) {
    double prob = 0.0;

    // Sum probabilities for values in window [x, x+winsize)
    // Convert 0-based window to 1-based Zipf values by adding 1
    size_t start_zipf = x + 1;         // Convert to 1-based
    size_t end_zipf = x + winsize + 1; // Convert to 1-based, exclusive end

    // OPTIMIZATION: Use precomputed values and accumulate sum first
    // PMF = k^(-s) / zeta(s), where zeta(s) is constant
    // So we can sum all k^(-s) first and divide once at the end
    for (size_t k = start_zipf; k < end_zipf && k <= max_k; k++) {
      prob += k_powers[k];
    }
    probabilities[x] = prob / zeta_value;
  }

  free(k_powers);
  return probabilities;
}
