#ifndef BROTLI_RESEARCH_DURCHSCHLAG_H_
#define BROTLI_RESEARCH_DURCHSCHLAG_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * Generate a dictionary for given samples.
 *
 * @param dictionary_size_limit maximal dictionary size
 * @param slice_len text slice size
 * @param block_len score block length
 * @param sample_sizes vector with sample sizes
 * @param sample_data concatenated samples
 * @return generated dictionary
 */
std::string durchschlag_generate(
    size_t dictionary_size_limit, size_t slice_len, size_t block_len,
    const std::vector<size_t>& sample_sizes, const uint8_t* sample_data);

//------------------------------------------------------------------------------
// Lower level API for repetitive dictionary generation.
//------------------------------------------------------------------------------

/* Pointer to position in text. */
typedef uint32_t DurchschlagTextIdx;

/* Context is made public for flexible serialization / deserialization. */
typedef struct DurchschlagContext {
  DurchschlagTextIdx dataSize;
  DurchschlagTextIdx sliceLen;
  DurchschlagTextIdx numUniqueSlices;
  std::vector<DurchschlagTextIdx> offsets;
  std::vector<DurchschlagTextIdx> sliceMap;
} DurchschlagContext;

DurchschlagContext durchschlag_prepare(size_t slice_len,
    const std::vector<size_t>& sample_sizes, const uint8_t* sample_data);

typedef enum DurchschalgResourceStrategy {
  // Faster
  DURCHSCHLAG_EXCLUSIVE = 0,
  // Uses much less memory
  DURCHSCHLAG_COLLABORATIVE = 1
} DurchschalgResourceStrategy;

std::string durchschlag_generate(DurchschalgResourceStrategy strategy,
    size_t dictionary_size_limit, size_t block_len,
    const DurchschlagContext& context, const uint8_t* sample_data);

//------------------------------------------------------------------------------
// Suffix Array based preparation.
//------------------------------------------------------------------------------

typedef struct DurchschlagIndex {
  std::vector<DurchschlagTextIdx> lcp;
  std::vector<DurchschlagTextIdx> sa;
} DurchschlagIndex;

DurchschlagIndex durchschlag_index(const std::vector<uint8_t>& data);

DurchschlagContext durchschlag_prepare(size_t slice_len,
    const std::vector<size_t>& sample_sizes, const DurchschlagIndex& index);

//------------------------------------------------------------------------------
// Data preparation.
//------------------------------------------------------------------------------

/**
 * Cut out unique slices.
 *
 * Both @p sample_sizes and @p sample_data are modified in-place. Number of
 * samples remains unchanged, but some samples become shorter.
 *
 * @param slice_len (unique) slice size
 * @param minimum_population minimum non-unique slice occurrence
 * @param sample_sizes [in / out] vector with sample sizes
 * @param sample_data [in / out] concatenated samples
 */
void durchschlag_distill(size_t slice_len, size_t minimum_population,
    std::vector<size_t>* sample_sizes, uint8_t* sample_data);

/**
 * Replace unique slices with zeroes.
 *
 * @p sample_data is modified in-place. Number of samples and their length
 * remain unchanged.
 *
 * @param slice_len (unique) slice size
 * @param minimum_population minimum non-unique slice occurrence
 * @param sample_sizes vector with sample sizes
 * @param sample_data [in / out] concatenated samples
 */
void durchschlag_purify(size_t slice_len, size_t minimum_population,
    const std::vector<size_t>& sample_sizes, uint8_t* sample_data);

#endif  // BROTLI_RESEARCH_DURCHSCHLAG_H_
