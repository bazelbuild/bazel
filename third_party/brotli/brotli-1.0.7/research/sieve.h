#ifndef BROTLI_RESEARCH_SIEVE_H_
#define BROTLI_RESEARCH_SIEVE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * Generate a dictionary for given samples.
 *
 * @param dictionary_size_limit maximal dictionary size
 * @param slice_len text slice size
 * @param sample_sizes vector with sample sizes
 * @param sample_data concatenated samples
 * @return generated dictionary
 */
std::string sieve_generate(size_t dictionary_size_limit, size_t slice_len,
    const std::vector<size_t>& sample_sizes, const uint8_t* sample_data);

#endif  // BROTLI_RESEARCH_SIEVE_H_
