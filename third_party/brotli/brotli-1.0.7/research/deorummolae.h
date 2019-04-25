#ifndef BROTLI_RESEARCH_DEORUMMOLAE_H_
#define BROTLI_RESEARCH_DEORUMMOLAE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/* log2(maximal number of files). Value 6 provides some speedups. */
#define DM_LOG_MAX_FILES 6

/* Non tunable definitions. */
#define DM_MAX_FILES (1 << DM_LOG_MAX_FILES)

/**
 * Generate a dictionary for given samples.
 *
 * @param dictionary_size_limit maximal dictionary size
 * @param sample_sizes vector with sample sizes
 * @param sample_data concatenated samples
 * @return generated dictionary
 */
std::string DM_generate(size_t dictionary_size_limit,
    const std::vector<size_t>& sample_sizes, const uint8_t* sample_data);

#endif  // BROTLI_RESEARCH_DEORUMMOLAE_H_
