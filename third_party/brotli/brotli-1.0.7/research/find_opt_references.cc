/* Copyright 2016 Google Inc. All Rights Reserved.
   Author: zip753@gmail.com (Ivan Nikulin)

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Tool for generating optimal backward references for the input file. Uses
   sais-lite library for building suffix array. */

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
using gflags::ParseCommandLineFlags;

#include "./esaxx/sais.hxx"

DEFINE_bool(advanced, false, "Advanced searching mode: finds all longest "
    "matches at positions that are not covered by matches of length at least "
    "max_length. WARNING: uses much more memory than simple mode, especially "
    "for small values of min_length.");
DEFINE_int32(min_length, 1, "Minimal length of found backward references.");
/* For advanced mode. */
DEFINE_int32(long_length, 32,
             "Maximal length of found backward references for advanced mode.");
DEFINE_int32(skip, 1, "Number of bytes to skip.");

const size_t kFileBufferSize = (1 << 16);  // 64KB

typedef int sarray_type;  // Can't make it unsigned because of templates :(
typedef uint8_t input_type;
typedef uint32_t lcp_type;
typedef std::pair<int, std::vector<int> > entry_type;
typedef std::function<void(sarray_type*, lcp_type*, size_t, int, int, int, int,
                           int)> Fn;

void ReadInput(FILE* fin, input_type* storage, size_t input_size) {
  size_t last_pos = 0;
  size_t available_in;
  fseek(fin, 0, SEEK_SET);
  do {
    available_in = fread(storage + last_pos, 1, kFileBufferSize, fin);
    last_pos += available_in;
  } while (available_in != 0);
  assert(last_pos == input_size);
}

void BuildLCP(input_type* storage, sarray_type* sarray, lcp_type* lcp,
              size_t size, uint32_t* pos) {
  for (int i = 0; i < size; ++i) {
    pos[sarray[i]] = i;
  }
  uint32_t k = 0;
  lcp[size - 1] = 0;
  for (int i = 0; i < size; ++i) {
    if (pos[i] == size - 1) {
      k = 0;
      continue;
    }
    uint32_t j = sarray[pos[i] + 1];  // Suffix which follow i-th suffix in SA.
    while (i + k < size && j + k < size && storage[i + k] == storage[j + k]) {
      ++k;
    }
    lcp[pos[i]] = k;
    if (k > 0) --k;
  }
}

inline void PrintReference(sarray_type* sarray, lcp_type* lcp, size_t size,
                           int idx, int left_ix, int right_ix, int left_lcp,
                           int right_lcp, FILE* fout) {
  int max_lcp_ix;
  if (right_ix == size - 1 || (left_ix >= 0 && left_lcp >= right_lcp)) {
    max_lcp_ix = left_ix;
  } else {
    max_lcp_ix = right_ix;
  }
  int dist = idx - sarray[max_lcp_ix];
  assert(dist > 0);
  fputc(1, fout);
  fwrite(&idx, sizeof(int), 1, fout);   // Position in input.
  fwrite(&dist, sizeof(int), 1, fout);  // Backward distance.
}

inline void GoLeft(sarray_type* sarray, lcp_type* lcp, int idx, int left_ix,
                   int left_lcp, entry_type* entry) {
  entry->first = left_lcp;
  if (left_lcp > FLAGS_long_length) return;
  for (; left_ix >= 0; --left_ix) {
    if (lcp[left_ix] < left_lcp) break;
    if (sarray[left_ix] < idx) {
      entry->second.push_back(idx - sarray[left_ix]);
    }
  }
}

inline void GoRight(sarray_type* sarray, lcp_type* lcp, int idx, size_t size,
                    int right_ix, int right_lcp, entry_type* entry) {
  entry->first = right_lcp;
  if (right_lcp > FLAGS_long_length) return;
  for (; right_ix < size - 1; ++right_ix) {
    if (lcp[right_ix] < right_lcp) break;
    if (sarray[right_ix] < idx) {
      entry->second.push_back(idx - sarray[right_ix]);
    }
  }
}

inline void StoreReference(sarray_type* sarray, lcp_type* lcp, size_t size,
                           int idx, int left_ix, int right_ix, int left_lcp,
                           int right_lcp, entry_type* entries) {
  if (right_ix == size - 1 || (left_ix >= 0 && left_lcp > right_lcp)) {
    // right is invalid or left is better
    GoLeft(sarray, lcp, idx, left_ix, left_lcp, &entries[idx]);
  } else if (left_ix < 0 || (right_ix < size - 1 && right_lcp > left_lcp)) {
    // left is invalid or right is better
    GoRight(sarray, lcp, idx, size, right_ix, right_lcp, &entries[idx]);
  } else {  // both are valid and of equal length
    GoLeft(sarray, lcp, idx, left_ix, left_lcp, &entries[idx]);
    GoRight(sarray, lcp, idx, size, right_ix, right_lcp, &entries[idx]);
  }
}

void ProcessReferences(sarray_type* sarray, lcp_type* lcp, size_t size,
                       uint32_t* pos, const Fn& process_output) {
  int min_length = FLAGS_min_length;
  for (int idx = FLAGS_skip; idx < size; ++idx) {
    int left_lcp = -1;
    int left_ix;
    for (left_ix = pos[idx] - 1; left_ix >= 0; --left_ix) {
      if (left_lcp == -1 || left_lcp > lcp[left_ix]) {
        left_lcp = lcp[left_ix];
      }
      if (left_lcp == 0) break;
      if (sarray[left_ix] < idx) break;
    }

    int right_lcp = -1;
    int right_ix;
    for (right_ix = pos[idx]; right_ix < size - 1; ++right_ix) {
      if (right_lcp == -1 || right_lcp > lcp[right_ix]) {
        right_lcp = lcp[right_ix];
      }
      // Stop if we have better result from the left side already.
      if (right_lcp < left_lcp && left_ix >= 0) break;
      if (right_lcp == 0) break;
      if (sarray[right_ix] < idx) break;
    }

    if ((left_ix >= 0 && left_lcp >= min_length) ||
        (right_ix < size - 1 && right_lcp >= min_length)) {
      process_output(sarray, lcp, size, idx, left_ix, right_ix, left_lcp,
                     right_lcp);
    }
  }
}

void ProcessEntries(entry_type* entries, size_t size, FILE* fout) {
  int long_length = FLAGS_long_length;
  std::vector<std::pair<int, int> > segments;
  size_t idx;
  for (idx = 0; idx < size;) {
    entry_type& entry = entries[idx];
    if (entry.first > long_length) {
      // Add segment.
      if (segments.empty() || segments.back().second < idx) {
        segments.push_back({idx, idx + entry.first});
      } else {
        segments.back().second = idx + entry.first;
      }
    }
    ++idx;
  }
  printf("Segments generated.\n");
  size_t segments_ix = 0;
  for (idx = 0; idx < size;) {
    if (idx == segments[segments_ix].first) {
      // Skip segment.
      idx = segments[segments_ix].second;
    } else {
      for (auto& dist : entries[idx].second) {
        fputc(1, fout);
        fwrite(&idx, sizeof(int), 1, fout);   // Position in input.
        fwrite(&dist, sizeof(int), 1, fout);  // Backward distance.
      }
      ++idx;
    }
  }
}

int main(int argc, char* argv[]) {
  ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 3) {
    printf("usage: %s input_file output_file\n", argv[0]);
    return 1;
  }

  FILE* fin = fopen(argv[1], "rb");
  FILE* fout = fopen(argv[2], "w");

  fseek(fin, 0, SEEK_END);
  int input_size = ftell(fin);
  fseek(fin, 0, SEEK_SET);
  printf("The file size is %u bytes\n", input_size);

  input_type* storage = new input_type[input_size];

  ReadInput(fin, storage, input_size);
  fclose(fin);

  sarray_type* sarray = new sarray_type[input_size];
  saisxx(storage, sarray, input_size);
  printf("Suffix array calculated.\n");

  // Inverse suffix array.
  uint32_t* pos = new uint32_t[input_size];

  lcp_type* lcp = new lcp_type[input_size];
  BuildLCP(storage, sarray, lcp, input_size, pos);
  printf("LCP array constructed.\n");
  delete[] storage;

  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  using std::placeholders::_4;
  using std::placeholders::_5;
  using std::placeholders::_6;
  using std::placeholders::_7;
  using std::placeholders::_8;
  entry_type* entries;
  if (FLAGS_advanced) {
    entries = new entry_type[input_size];
    for (size_t i = 0; i < input_size; ++i) entries[i].first = -1;
  }
  Fn print = std::bind(PrintReference, _1, _2, _3, _4, _5, _6, _7, _8, fout);
  Fn store = std::bind(StoreReference, _1, _2, _3, _4, _5, _6, _7, _8, entries);

  ProcessReferences(sarray, lcp, input_size, pos,
                    FLAGS_advanced ? store : print);
  printf("References processed.\n");

  if (FLAGS_advanced) {
    int good_cnt = 0;
    uint64_t avg_cnt = 0;
    for (size_t i = 0; i < input_size; ++i) {
      if (entries[i].first != -1) {
        ++good_cnt;
        avg_cnt += entries[i].second.size();
      }
    }
    printf("Number of covered positions = %d\n", good_cnt);
    printf("Average number of references per covered position = %.4lf\n",
            static_cast<double>(avg_cnt) / good_cnt);
    ProcessEntries(entries, input_size, fout);
    printf("Entries processed.\n");
  }

  fclose(fout);
  return 0;
}
