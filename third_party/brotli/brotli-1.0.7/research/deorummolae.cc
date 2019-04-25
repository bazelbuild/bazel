#include "./deorummolae.h"

#include <array>
#include <cstdio>

#include "./esaxx/sais.hxx"

/* Used for quick SA-entry to file mapping. Each file is padded to size that
   is a multiple of chunk size. */
#define CHUNK_SIZE 64
/* Length of substring that is considered to be covered by dictionary string. */
#define CUT_MATCH 6
/* Minimal dictionary entry size. */
#define MIN_MATCH 24

/* Non tunable definitions. */
#define CHUNK_MASK (CHUNK_SIZE - 1)
#define COVERAGE_SIZE (1 << (DM_LOG_MAX_FILES - 6))

/* File coverage: every bit set to 1 denotes a file covered by an isle. */
typedef std::array<uint64_t, COVERAGE_SIZE> Coverage;

/* Symbol of text alphabet. */
typedef int32_t TextChar;

/* Pointer to position in text. */
typedef uint32_t TextIdx;

/* SAIS sarray_type; unfortunately, must be a signed type. */
typedef int32_t TextSaIdx;

static size_t popcount(uint64_t u) {
  return static_cast<size_t>(__builtin_popcountll(u));
}

/* Condense terminators and pad file entries. */
static void rewriteText(std::vector<TextChar>* text) {
  TextChar terminator = text->back();
  TextChar prev = terminator;
  TextIdx to = 0;
  for (TextIdx from = 0; from < text->size(); ++from) {
    TextChar next = text->at(from);
    if (next < 256 || prev < 256) {
      text->at(to++) = next;
      if (next >= 256) terminator = next;
    }
    prev = next;
  }
  text->resize(to);
  if (text->empty()) text->push_back(terminator);
  while (text->size() & CHUNK_MASK) text->push_back(terminator);
}

/* Reenumerate terminators for smaller alphabet. */
static void remapTerminators(std::vector<TextChar>* text,
    TextChar* next_terminator) {
  TextChar prev = -1;
  TextChar x = 256;
  for (TextIdx i = 0; i < text->size(); ++i) {
    TextChar next = text->at(i);
    if (next < 256) {  // Char.
      // Do nothing.
    } else if (prev < 256) {  // Terminator after char.
      next = x++;
    } else {  // Terminator after terminator.
      next = prev;
    }
    text->at(i) = next;
    prev = next;
  }
  *next_terminator = x;
}

/* Combine all file entries; create mapping position->file. */
static void buildFullText(std::vector<std::vector<TextChar>>* data,
    std::vector<TextChar>* full_text, std::vector<TextIdx>* file_map,
    std::vector<TextIdx>* file_offset, TextChar* next_terminator) {
  file_map->resize(0);
  file_offset->resize(0);
  full_text->resize(0);
  for (TextIdx i = 0; i < data->size(); ++i) {
    file_offset->push_back(full_text->size());
    std::vector<TextChar>& file = data->at(i);
    rewriteText(&file);
    full_text->insert(full_text->end(), file.begin(), file.end());
    file_map->insert(file_map->end(), file.size() / CHUNK_SIZE, i);
  }
  if (false) remapTerminators(full_text, next_terminator);
}

/* Build longest-common-prefix based on suffix array and text.
   TODO: borrowed -> unknown efficiency. */
static void buildLcp(std::vector<TextChar>* text, std::vector<TextIdx>* sa,
    std::vector<TextIdx>* lcp, std::vector<TextIdx>* invese_sa) {
  TextIdx size = static_cast<TextIdx>(text->size());
  lcp->resize(size);
  TextIdx k = 0;
  lcp->at(size - 1) = 0;
  for (TextIdx i = 0; i < size; ++i) {
    if (invese_sa->at(i) == size - 1) {
      k = 0;
      continue;
    }
    // Suffix which follow i-th suffix.
    TextIdx j = sa->at(invese_sa->at(i) + 1);
    while (i + k < size && j + k < size && text->at(i + k) == text->at(j + k)) {
      ++k;
    }
    lcp->at(invese_sa->at(i)) = k;
    if (k > 0) --k;
  }
}

/* Isle is a range in SA with LCP not less than some value.
   When we raise the LCP requirement, the isle sunks and smaller isles appear
   instead. */
typedef struct {
  TextIdx lcp;
  TextIdx l;
  TextIdx r;
  Coverage coverage;
} Isle;

/* Helper routine for `cutMatch`. */
static void poisonData(TextIdx pos, TextIdx length,
    std::vector<std::vector<TextChar>>* data, std::vector<TextIdx>* file_map,
    std::vector<TextIdx>* file_offset, TextChar* next_terminator) {
  TextIdx f = file_map->at(pos / CHUNK_SIZE);
  pos -= file_offset->at(f);
  std::vector<TextChar>& file = data->at(f);
  TextIdx l = (length == CUT_MATCH) ? CUT_MATCH : 1;
  for (TextIdx j = 0; j < l; j++, pos++) {
    if (file[pos] >= 256) continue;
    if (file[pos + 1] >= 256) {
      file[pos] = file[pos + 1];
    } else if (pos > 0 && file[pos - 1] >= 256) {
      file[pos] = file[pos - 1];
    } else {
      file[pos] = (*next_terminator)++;
    }
  }
}

/* Remove substrings of a given match from files.
   Substrings are replaced with unique terminators, so next iteration SA would
   not allow to cross removed areas. */
static void cutMatch(std::vector<std::vector<TextChar>>* data, TextIdx index,
    TextIdx length, std::vector<TextIdx>* sa, std::vector<TextIdx>* lcp,
    std::vector<TextIdx>* invese_sa, TextChar* next_terminator,
    std::vector<TextIdx>* file_map, std::vector<TextIdx>* file_offset) {
  while (length >= CUT_MATCH) {
    TextIdx i = index;
    while (lcp->at(i) >= length) {
      i++;
      poisonData(
          sa->at(i), length, data, file_map, file_offset, next_terminator);
    }
    while (true) {
      poisonData(
          sa->at(index), length, data, file_map, file_offset, next_terminator);
      if (index == 0 || lcp->at(index - 1) < length) break;
      index--;
    }
    length--;
    index = invese_sa->at(sa->at(index) + 1);
  }
}

std::string DM_generate(size_t dictionary_size_limit,
    const std::vector<size_t>& sample_sizes, const uint8_t* sample_data) {
  {
    TextIdx tmp = static_cast<TextIdx>(dictionary_size_limit);
    if ((tmp != dictionary_size_limit) || (tmp > 1u << 30)) {
      fprintf(stderr, "dictionary_size_limit is too large\n");
      return "";
    }
  }

  /* Could use 256 + '0' for easier debugging. */
  TextChar next_terminator = 256;

  std::string output;
  std::vector<std::vector<TextChar>> data;

  TextIdx offset = 0;
  size_t num_samples = sample_sizes.size();
  if (num_samples > DM_MAX_FILES) num_samples = DM_MAX_FILES;
  for (size_t n = 0; n < num_samples; ++n) {
    TextIdx delta = static_cast<TextIdx>(sample_sizes[n]);
    if (delta != sample_sizes[n]) {
      fprintf(stderr, "sample is too large\n");
      return "";
    }
    if (delta == 0) {
      fprintf(stderr, "0-length samples are prohibited\n");
      return "";
    }
    TextIdx next_offset = offset + delta;
    if (next_offset <= offset) {
      fprintf(stderr, "corpus is too large\n");
      return "";
    }
    data.push_back(
        std::vector<TextChar>(sample_data + offset, sample_data + next_offset));
    offset = next_offset;
    data.back().push_back(next_terminator++);
  }

  /* Most arrays are allocated once, and then just resized to smaller and
     smaller sizes. */
  std::vector<TextChar> full_text;
  std::vector<TextIdx> file_map;
  std::vector<TextIdx> file_offset;
  std::vector<TextIdx> sa;
  std::vector<TextIdx> invese_sa;
  std::vector<TextIdx> lcp;
  std::vector<Isle> isles;
  std::vector<char> output_data;
  TextIdx total = 0;
  TextIdx total_cost = 0;
  TextIdx best_cost;
  Isle best_isle;
  size_t min_count = num_samples;

  while (true) {
    TextIdx max_match = static_cast<TextIdx>(dictionary_size_limit) - total;
    buildFullText(&data, &full_text, &file_map, &file_offset, &next_terminator);
    sa.resize(full_text.size());
    /* Hopefully, non-negative TextSaIdx is the same sa TextIdx counterpart. */
    saisxx(full_text.data(), reinterpret_cast<TextSaIdx*>(sa.data()),
        static_cast<TextChar>(full_text.size()), next_terminator);
    invese_sa.resize(full_text.size());
    for (TextIdx i = 0; i < full_text.size(); ++i) {
      invese_sa[sa[i]] = i;
    }
    buildLcp(&full_text, &sa, &lcp, &invese_sa);

    /* Do not rebuild SA/LCP, just use different selection. */
  retry:
    best_cost = 0;
    best_isle = {0, 0, 0, {{0}}};
    isles.resize(0);
    isles.push_back(best_isle);

    for (TextIdx i = 0; i < lcp.size(); ++i) {
      TextIdx l = i;
      Coverage cov = {{0}};
      size_t f = file_map[sa[i] / CHUNK_SIZE];
      cov[f >> 6] = (static_cast<uint64_t>(1)) << (f & 63);
      while (lcp[i] < isles.back().lcp) {
        Isle& top = isles.back();
        top.r = i;
        l = top.l;
        for (size_t x = 0; x < cov.size(); ++x) cov[x] |= top.coverage[x];
        size_t count = 0;
        for (size_t x = 0; x < cov.size(); ++x) count += popcount(cov[x]);
        TextIdx effective_lcp = top.lcp;
        /* Restrict (last) dictionary entry length. */
        if (effective_lcp > max_match) effective_lcp = max_match;
        TextIdx cost = count * effective_lcp;
        if (cost > best_cost && count >= min_count &&
            effective_lcp >= MIN_MATCH) {
          best_cost = cost;
          best_isle = top;
          best_isle.lcp = effective_lcp;
        }
        isles.pop_back();
        for (size_t x = 0; x < cov.size(); ++x) {
          isles.back().coverage[x] |= cov[x];
        }
      }
      if (lcp[i] > isles.back().lcp) isles.push_back({lcp[i], l, 0, {{0}}});
      for (size_t x = 0; x < cov.size(); ++x) {
        isles.back().coverage[x] |= cov[x];
      }
    }

    /* When saturated matches do not match length restrictions, lower the
       saturation requirements. */
    if (best_cost == 0 || best_isle.lcp < MIN_MATCH) {
      if (min_count >= 8) {
        min_count = (min_count * 7) / 8;
        fprintf(stderr, "Retry: min_count=%zu\n", min_count);
        goto retry;
      }
      break;
    }

    /* Save the entry. */
    fprintf(stderr, "Savings: %d+%d, dictionary: %d+%d\n",
        total_cost, best_cost, total, best_isle.lcp);
    int* piece = &full_text[sa[best_isle.l]];
    output.insert(output.end(), piece, piece + best_isle.lcp);
    total += best_isle.lcp;
    total_cost += best_cost;
    cutMatch(&data, best_isle.l, best_isle.lcp, &sa, &lcp, &invese_sa,
        &next_terminator, &file_map, &file_offset);
    if (total >= dictionary_size_limit) break;
  }

  return output;
}
