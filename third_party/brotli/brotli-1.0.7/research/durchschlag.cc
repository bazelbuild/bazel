#include "./durchschlag.h"

#include <algorithm>
#include <exception>  /* terminate */

#include "divsufsort.h"

/* Pointer to position in text. */
typedef DurchschlagTextIdx TextIdx;

/* (Sum of) value(s) of slice(s). */
typedef uint32_t Score;

typedef struct HashSlot {
  TextIdx next;
  TextIdx offset;
} HashSlot;

typedef struct MetaSlot {
  TextIdx mark;
  Score score;
} MetaSlot;

typedef struct Range {
  TextIdx start;
  TextIdx end;
} Range;

typedef struct Candidate {
  Score score;
  TextIdx position;
} Candidate;

struct greaterScore {
  bool operator()(const Candidate& a, const Candidate& b) const {
    return (a.score > b.score) ||
        ((a.score == b.score) && (a.position < b.position));
  }
};

struct lessScore {
  bool operator()(const Candidate& a, const Candidate& b) const {
    return (a.score < b.score) ||
        ((a.score == b.score) && (a.position > b.position));
  }
};

#define CANDIDATE_BUNDLE_SIZE (1 << 18)

static void fatal(const char* error) {
  fprintf(stderr, "%s\n", error);
  std::terminate();
}

static TextIdx calculateDictionarySize(const std::vector<Range>& ranges) {
  TextIdx result = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const Range& r = ranges[i];
    result += r.end - r.start;
  }
  return result;
}

static std::string createDictionary(
    const uint8_t* data, const std::vector<Range>& ranges, size_t limit) {
  std::string output;
  output.reserve(calculateDictionarySize(ranges));
  for (size_t i = 0; i < ranges.size(); ++i) {
    const Range& r = ranges[i];
    output.insert(output.end(), &data[r.start], &data[r.end]);
  }
  if (output.size() > limit) {
    output.resize(limit);
  }
  return output;
}

/* precondition: span > 0
   precondition: end + span == len(shortcut) */
static Score buildCandidatesList(std::vector<Candidate>* candidates,
    std::vector<MetaSlot>* map, TextIdx span, const TextIdx* shortcut,
    TextIdx end) {
  candidates->resize(0);

  size_t n = map->size();
  MetaSlot* slots = map->data();
  for (size_t j = 0; j < n; ++j) {
    slots[j].mark = 0;
  }

  Score score = 0;
  /* Consider the whole span, except one last item. The following loop will
     add the last item to the end of the "chain", evaluate it, and cut one
     "link" form the beginning. */
  for (size_t j = 0; j < span - 1; ++j) {
    MetaSlot& item = slots[shortcut[j]];
    if (item.mark == 0) {
      score += item.score;
    }
    item.mark++;
  }

  TextIdx i = 0;
  TextIdx limit = std::min<TextIdx>(end, CANDIDATE_BUNDLE_SIZE);
  Score maxScore = 0;
  for (; i < limit; ++i) {
    TextIdx slice = shortcut[i + span - 1];
    MetaSlot& pick = slots[slice];
    if (pick.mark == 0) {
      score += pick.score;
    }
    pick.mark++;

    if (score > maxScore) {
      maxScore = score;
    }
    candidates->push_back({score, i});

    MetaSlot& drop = slots[shortcut[i]];
    drop.mark--;
    if (drop.mark == 0) {
      score -= drop.score;
    }
  }

  std::make_heap(candidates->begin(), candidates->end(), greaterScore());
  Score minScore = candidates->at(0).score;
  for (; i < end; ++i) {
    TextIdx slice = shortcut[i + span - 1];
    MetaSlot& pick = slots[slice];
    if (pick.mark == 0) {
      score += pick.score;
    }
    pick.mark++;

    if (score > maxScore) {
      maxScore = score;
    }
    if (score >= minScore) {
      candidates->push_back({score, i});
      std::push_heap(candidates->begin(), candidates->end(), greaterScore());
      if (candidates->size() > CANDIDATE_BUNDLE_SIZE && maxScore != minScore) {
        while (candidates->at(0).score == minScore) {
          std::pop_heap(candidates->begin(), candidates->end(), greaterScore());
          candidates->pop_back();
        }
        minScore = candidates->at(0).score;
      }
    }

    MetaSlot& drop = slots[shortcut[i]];
    drop.mark--;
    if (drop.mark == 0) {
      score -= drop.score;
    }
  }

  for (size_t j = 0; j < n; ++j) {
    slots[j].mark = 0;
  }

  std::make_heap(candidates->begin(), candidates->end(), lessScore());
  return minScore;
}

/* precondition: span > 0
   precondition: end + span == len(shortcut) */
static Score rebuildCandidatesList(std::vector<TextIdx>* candidates,
    std::vector<MetaSlot>* map, TextIdx span, const TextIdx* shortcut,
    TextIdx end, TextIdx* next) {
  size_t n = candidates->size();
  TextIdx* data = candidates->data();
  for (size_t i = 0; i < n; ++i) {
    data[i] = 0;
  }

  n = map->size();
  MetaSlot* slots = map->data();
  for (size_t i = 0; i < n; ++i) {
    slots[i].mark = 0;
  }

  Score score = 0;
  /* Consider the whole span, except one last item. The following loop will
     add the last item to the end of the "chain", evaluate it, and cut one
     "link" form the beginning. */
  for (TextIdx i = 0; i < span - 1; ++i) {
    MetaSlot& item = slots[shortcut[i]];
    if (item.mark == 0) {
      score += item.score;
    }
    item.mark++;
  }

  Score maxScore = 0;
  for (TextIdx i = 0; i < end; ++i) {
    MetaSlot& pick = slots[shortcut[i + span - 1]];
    if (pick.mark == 0) {
      score += pick.score;
    }
    pick.mark++;

    if (candidates->size() <= score) {
      candidates->resize(score + 1);
    }
    if (score > maxScore) {
      maxScore = score;
    }
    next[i] = candidates->at(score);
    candidates->at(score) = i;

    MetaSlot& drop = slots[shortcut[i]];
    drop.mark--;
    if (drop.mark == 0) {
      score -= drop.score;
    }
  }

  for (size_t i = 0; i < n; ++i) {
    slots[i].mark = 0;
  }

  candidates->resize(maxScore + 1);
  return maxScore;
}

static void addRange(std::vector<Range>* ranges, TextIdx start, TextIdx end) {
  for (auto it = ranges->begin(); it != ranges->end();) {
    if (end < it->start) {
      ranges->insert(it, {start, end});
      return;
    }
    if (it->end < start) {
      it++;
      continue;
    }
    // Combine with existing.
    start = std::min(start, it->start);
    end = std::max(end, it->end);
    // Remove consumed vector and continue.
    it = ranges->erase(it);
  }
  ranges->push_back({start, end});
}

std::string durchschlag_generate(
    size_t dictionary_size_limit, size_t slice_len, size_t block_len,
    const std::vector<size_t>& sample_sizes, const uint8_t* sample_data) {
  DurchschlagContext ctx = durchschlag_prepare(
      slice_len, sample_sizes, sample_data);
  return durchschlag_generate(DURCHSCHLAG_COLLABORATIVE,
      dictionary_size_limit, block_len, ctx, sample_data);
}

DurchschlagContext durchschlag_prepare(size_t slice_len,
    const std::vector<size_t>& sample_sizes, const uint8_t* sample_data) {
  /* Parameters aliasing */
  TextIdx sliceLen = static_cast<TextIdx>(slice_len);
  if (sliceLen != slice_len) fatal("slice_len is too large");
  if (sliceLen < 1) fatal("slice_len is too small");
  const uint8_t* data = sample_data;

  TextIdx total = 0;
  std::vector<TextIdx> offsets;
  offsets.reserve(sample_sizes.size());
  for (size_t i = 0; i < sample_sizes.size(); ++i) {
    TextIdx delta = static_cast<TextIdx>(sample_sizes[i]);
    if (delta != sample_sizes[i]) fatal("sample is too large");
    if (delta == 0) fatal("0-length samples are prohibited");
    TextIdx next_total = total + delta;
    if (next_total <= total) fatal("corpus is too large");
    total = next_total;
    offsets.push_back(total);
  }

  if (total < sliceLen) fatal("slice_len is larger than corpus size");
  TextIdx end = total - static_cast<TextIdx>(sliceLen) + 1;
  TextIdx hashLen = 11;
  while (hashLen < 29 && ((1u << hashLen) < end)) {
    hashLen += 3;
  }
  hashLen -= 3;
  TextIdx hashMask = (1u << hashLen) - 1u;
  std::vector<TextIdx> hashHead(1 << hashLen);
  TextIdx hash = 0;
  TextIdx lShift = 3;
  TextIdx rShift = hashLen - lShift;
  for (TextIdx i = 0; i < sliceLen - 1; ++i) {
    TextIdx v = data[i];
    hash = (((hash << lShift) | (hash >> rShift)) & hashMask) ^ v;
  }
  TextIdx lShiftX = (lShift * (sliceLen - 1)) % hashLen;
  TextIdx rShiftX = hashLen - lShiftX;

  std::vector<HashSlot> map;
  map.push_back({0, 0});
  TextIdx hashSlot = 1;
  std::vector<TextIdx> sliceMap;
  sliceMap.reserve(end);
  for (TextIdx i = 0; i < end; ++i) {
    TextIdx v = data[i + sliceLen - 1];
    TextIdx bucket = (((hash << lShift) | (hash >> rShift)) & hashMask) ^ v;
    v = data[i];
    hash = bucket ^ (((v << lShiftX) | (v >> rShiftX)) & hashMask);
    TextIdx slot = hashHead[bucket];
    while (slot != 0) {
      HashSlot& item = map[slot];
      TextIdx start = item.offset;
      bool miss = false;
      for (TextIdx j = 0; j < sliceLen; ++j) {
        if (data[i + j] != data[start + j]) {
          miss = true;
          break;
        }
      }
      if (!miss) {
        sliceMap.push_back(slot);
        break;
      }
      slot = item.next;
    }
    if (slot == 0) {
      map.push_back({hashHead[bucket], i});
      hashHead[bucket] = hashSlot;
      sliceMap.push_back(hashSlot);
      hashSlot++;
    }
  }

  return {total, sliceLen, static_cast<TextIdx>(map.size()),
      std::move(offsets), std::move(sliceMap)};
}

DurchschlagContext durchschlag_prepare(size_t slice_len,
    const std::vector<size_t>& sample_sizes, const DurchschlagIndex& index) {
  /* Parameters aliasing */
  TextIdx sliceLen = static_cast<TextIdx>(slice_len);
  if (sliceLen != slice_len) fatal("slice_len is too large");
  if (sliceLen < 1) fatal("slice_len is too small");
  const TextIdx* lcp = index.lcp.data();
  const TextIdx* sa = index.sa.data();

  TextIdx total = 0;
  std::vector<TextIdx> offsets;
  offsets.reserve(sample_sizes.size());
  for (size_t i = 0; i < sample_sizes.size(); ++i) {
    TextIdx delta = static_cast<TextIdx>(sample_sizes[i]);
    if (delta != sample_sizes[i]) fatal("sample is too large");
    if (delta == 0) fatal("0-length samples are prohibited");
    TextIdx next_total = total + delta;
    if (next_total <= total) fatal("corpus is too large");
    total = next_total;
    offsets.push_back(total);
  }

  if (total < sliceLen) fatal("slice_len is larger than corpus size");
  TextIdx counter = 1;
  TextIdx end = total - sliceLen + 1;
  std::vector<TextIdx> sliceMap(total);
  TextIdx last = 0;
  TextIdx current = 1;
  while (current <= total) {
    if (lcp[current - 1] < sliceLen) {
      for (TextIdx i = last; i < current; ++i) {
        sliceMap[sa[i]] = counter;
      }
      counter++;
      last = current;
    }
    current++;
  }
  sliceMap.resize(end);

  // Reorder items for the better locality.
  std::vector<TextIdx> reorder(counter);
  counter = 1;
  for (TextIdx i = 0; i < end; ++i) {
    if (reorder[sliceMap[i]] == 0) {
      reorder[sliceMap[i]] = counter++;
    }
  }
  for (TextIdx i = 0; i < end; ++i) {
    sliceMap[i] = reorder[sliceMap[i]];
  }

  return {total, sliceLen, counter, std::move(offsets), std::move(sliceMap)};
}

DurchschlagIndex durchschlag_index(const std::vector<uint8_t>& data) {
  TextIdx total = static_cast<TextIdx>(data.size());
  if (total != data.size()) fatal("corpus is too large");
  saidx_t saTotal = static_cast<saidx_t>(total);
  if (saTotal < 0) fatal("corpus is too large");
  if (static_cast<TextIdx>(saTotal) != total) fatal("corpus is too large");
  std::vector<TextIdx> sa(total);
  /* Hopefully, non-negative int32_t values match TextIdx ones. */
  if (sizeof(TextIdx) != sizeof(int32_t)) fatal("type length mismatch");
  int32_t* saData = reinterpret_cast<int32_t*>(sa.data());
  divsufsort(data.data(), saData, saTotal);

  std::vector<TextIdx> isa(total);
  for (TextIdx i = 0; i < total; ++i) isa[sa[i]] = i;

  // TODO: borrowed -> unknown efficiency.
  std::vector<TextIdx> lcp(total);
  TextIdx k = 0;
  lcp[total - 1] = 0;
  for (TextIdx i = 0; i < total; ++i) {
    TextIdx current = isa[i];
    if (current == total - 1) {
      k = 0;
      continue;
    }
    TextIdx j = sa[current + 1];  // Suffix which follow i-th suffix.
    while ((i + k < total) && (j + k < total) && (data[i + k] == data[j + k])) {
      ++k;
    }
    lcp[current] = k;
    if (k > 0) --k;
  }

  return {std::move(lcp), std::move(sa)};
}

static void ScoreSlices(const std::vector<TextIdx>& offsets,
    std::vector<MetaSlot>& map, const TextIdx* shortcut, TextIdx end) {
  TextIdx piece = 0;
  /* Fresh map contains all zeroes -> initial mark should be different. */
  TextIdx mark = 1;
  for (TextIdx i = 0; i < end; ++i) {
    if (offsets[piece] == i) {
      piece++;
      mark++;
    }
    MetaSlot& item = map[shortcut[i]];
    if (item.mark != mark) {
      item.mark = mark;
      item.score++;
    }
  }
}

static std::string durchschlagGenerateExclusive(
    size_t dictionary_size_limit, size_t block_len,
    const DurchschlagContext& context, const uint8_t* sample_data) {
  /* Parameters aliasing */
  TextIdx targetSize = static_cast<TextIdx>(dictionary_size_limit);
  if (targetSize != dictionary_size_limit) {
    fprintf(stderr, "dictionary_size_limit is too large\n");
    return "";
  }
  TextIdx sliceLen = context.sliceLen;
  TextIdx total = context.dataSize;
  TextIdx blockLen = static_cast<TextIdx>(block_len);
  if (blockLen != block_len) {
    fprintf(stderr, "block_len is too large\n");
    return "";
  }
  const uint8_t* data = sample_data;
  const std::vector<TextIdx>& offsets = context.offsets;
  std::vector<MetaSlot> map(context.numUniqueSlices);
  const TextIdx* shortcut = context.sliceMap.data();

  /* Initialization */
  if (blockLen < sliceLen) {
    fprintf(stderr, "sliceLen is larger than block_len\n");
    return "";
  }
  if (targetSize < blockLen || total < blockLen) {
    fprintf(stderr, "block_len is too large\n");
    return "";
  }
  TextIdx end = total - sliceLen + 1;
  ScoreSlices(offsets, map, shortcut, end);
  TextIdx span = blockLen - sliceLen + 1;
  end = static_cast<TextIdx>(context.sliceMap.size()) - span;
  std::vector<TextIdx> candidates;
  std::vector<TextIdx> next(end);
  Score maxScore = rebuildCandidatesList(
      &candidates, &map, span, shortcut, end, next.data());

  /* Block selection */
  const size_t triesLimit = (600 * 1000000) / span;
  const size_t candidatesLimit = (150 * 1000000) / span;
  std::vector<Range> ranges;
  TextIdx mark = 0;
  size_t numTries = 0;
  while (true) {
    TextIdx dictSize = calculateDictionarySize(ranges);
    size_t numCandidates = 0;
    if (dictSize > targetSize - blockLen) {
      break;
    }
    if (maxScore == 0) {
      break;
    }
    while (true) {
      TextIdx candidate = 0;
      while (maxScore > 0) {
        if (candidates[maxScore] != 0) {
          candidate = candidates[maxScore];
          candidates[maxScore] = next[candidate];
          break;
        }
        maxScore--;
      }
      if (maxScore == 0) {
        break;
      }
      mark++;
      numTries++;
      numCandidates++;
      Score score = 0;
      for (size_t j = candidate; j < candidate + span; ++j) {
        MetaSlot& item = map[shortcut[j]];
        if (item.mark != mark) {
          score += item.score;
          item.mark = mark;
        }
      }
      if (score < maxScore) {
        if (numTries < triesLimit && numCandidates < candidatesLimit) {
          next[candidate] = candidates[score];
          candidates[score] = candidate;
        } else {
          maxScore = rebuildCandidatesList(
              &candidates, &map, span, shortcut, end, next.data());
          mark = 0;
          numTries = 0;
          numCandidates = 0;
        }
        continue;
      } else if (score > maxScore) {
        fprintf(stderr, "Broken invariant\n");
        return "";
      }
      for (TextIdx j = candidate; j < candidate + span; ++j) {
        MetaSlot& item = map[shortcut[j]];
        item.score = 0;
      }
      addRange(&ranges, candidate, candidate + blockLen);
      break;
    }
  }

  return createDictionary(data, ranges, targetSize);
}

static std::string durchschlagGenerateCollaborative(
    size_t dictionary_size_limit, size_t block_len,
    const DurchschlagContext& context, const uint8_t* sample_data) {
  /* Parameters aliasing */
  TextIdx targetSize = static_cast<TextIdx>(dictionary_size_limit);
  if (targetSize != dictionary_size_limit) {
    fprintf(stderr, "dictionary_size_limit is too large\n");
    return "";
  }
  TextIdx sliceLen = context.sliceLen;
  TextIdx total = context.dataSize;
  TextIdx blockLen = static_cast<TextIdx>(block_len);
  if (blockLen != block_len) {
    fprintf(stderr, "block_len is too large\n");
    return "";
  }
  const uint8_t* data = sample_data;
  const std::vector<TextIdx>& offsets = context.offsets;
  std::vector<MetaSlot> map(context.numUniqueSlices);
  const TextIdx* shortcut = context.sliceMap.data();

  /* Initialization */
  if (blockLen < sliceLen) {
    fprintf(stderr, "sliceLen is larger than block_len\n");
    return "";
  }
  if (targetSize < blockLen || total < blockLen) {
    fprintf(stderr, "block_len is too large\n");
    return "";
  }
  TextIdx end = total - sliceLen + 1;
  ScoreSlices(offsets, map, shortcut, end);
  TextIdx span = blockLen - sliceLen + 1;
  end = static_cast<TextIdx>(context.sliceMap.size()) - span;
  std::vector<Candidate> candidates;
  candidates.reserve(CANDIDATE_BUNDLE_SIZE + 1024);
  Score minScore = buildCandidatesList(&candidates, &map, span, shortcut, end);

  /* Block selection */
  std::vector<Range> ranges;
  TextIdx mark = 0;
  while (true) {
    TextIdx dictSize = calculateDictionarySize(ranges);
    if (dictSize > targetSize - blockLen) {
      break;
    }
    if (minScore == 0 && candidates.empty()) {
      break;
    }
    while (true) {
      if (candidates.empty()) {
        minScore = buildCandidatesList(&candidates, &map, span, shortcut, end);
        mark = 0;
      }
      TextIdx candidate = candidates[0].position;
      Score expectedScore = candidates[0].score;
      if (expectedScore == 0) {
        candidates.resize(0);
        break;
      }
      std::pop_heap(candidates.begin(), candidates.end(), lessScore());
      candidates.pop_back();
      mark++;
      Score score = 0;
      for (TextIdx j = candidate; j < candidate + span; ++j) {
        MetaSlot& item = map[shortcut[j]];
        if (item.mark != mark) {
          score += item.score;
          item.mark = mark;
        }
      }
      if (score < expectedScore) {
        if (score >= minScore) {
          candidates.push_back({score, candidate});
          std::push_heap(candidates.begin(), candidates.end(), lessScore());
        }
        continue;
      } else if (score > expectedScore) {
        fatal("Broken invariant");
      }
      for (TextIdx j = candidate; j < candidate + span; ++j) {
        MetaSlot& item = map[shortcut[j]];
        item.score = 0;
      }
      addRange(&ranges, candidate, candidate + blockLen);
      break;
    }
  }

  return createDictionary(data, ranges, targetSize);
}

std::string durchschlag_generate(DurchschalgResourceStrategy strategy,
    size_t dictionary_size_limit, size_t block_len,
    const DurchschlagContext& context, const uint8_t* sample_data) {
  if (strategy == DURCHSCHLAG_COLLABORATIVE) {
    return durchschlagGenerateCollaborative(
        dictionary_size_limit, block_len, context, sample_data);
  } else {
    return durchschlagGenerateExclusive(
        dictionary_size_limit, block_len, context, sample_data);
  }
}

void durchschlag_distill(size_t slice_len, size_t minimum_population,
    std::vector<size_t>* sample_sizes, uint8_t* sample_data) {
  /* Parameters aliasing */
  uint8_t* data = sample_data;

  /* Build slice map. */
  DurchschlagContext context = durchschlag_prepare(
      slice_len, *sample_sizes, data);

  /* Calculate slice population. */
  const std::vector<TextIdx>& offsets = context.offsets;
  std::vector<MetaSlot> map(context.numUniqueSlices);
  const TextIdx* shortcut = context.sliceMap.data();
  TextIdx sliceLen = context.sliceLen;
  TextIdx total = context.dataSize;
  TextIdx end = total - sliceLen + 1;
  ScoreSlices(offsets, map, shortcut, end);

  /* Condense samples, omitting unique slices. */
  TextIdx readPos = 0;
  TextIdx writePos = 0;
  TextIdx lastNonUniquePos = 0;
  for (TextIdx i = 0; i < sample_sizes->size(); ++i) {
    TextIdx sampleStart = writePos;
    TextIdx oldSampleEnd =
        readPos + static_cast<TextIdx>(sample_sizes->at(i));
    while (readPos < oldSampleEnd) {
      if (readPos < end) {
        MetaSlot& item = map[shortcut[readPos]];
        if (item.score >= minimum_population) {
          lastNonUniquePos = readPos + sliceLen;
        }
      }
      if (readPos < lastNonUniquePos) {
        data[writePos++] = data[readPos];
      }
      readPos++;
    }
    sample_sizes->at(i) = writePos - sampleStart;
  }
}

void durchschlag_purify(size_t slice_len, size_t minimum_population,
    const std::vector<size_t>& sample_sizes, uint8_t* sample_data) {
  /* Parameters aliasing */
  uint8_t* data = sample_data;

  /* Build slice map. */
  DurchschlagContext context = durchschlag_prepare(
      slice_len, sample_sizes, data);

  /* Calculate slice population. */
  const std::vector<TextIdx>& offsets = context.offsets;
  std::vector<MetaSlot> map(context.numUniqueSlices);
  const TextIdx* shortcut = context.sliceMap.data();
  TextIdx sliceLen = context.sliceLen;
  TextIdx total = context.dataSize;
  TextIdx end = total - sliceLen + 1;
  ScoreSlices(offsets, map, shortcut, end);

  /* Rewrite samples, zeroing out unique slices. */
  TextIdx lastNonUniquePos = 0;
  for (TextIdx readPos = 0; readPos < total; ++readPos) {
    if (readPos < end) {
      MetaSlot& item = map[shortcut[readPos]];
      if (item.score >= minimum_population) {
        lastNonUniquePos = readPos + sliceLen;
      }
    }
    if (readPos >= lastNonUniquePos) {
      data[readPos] = 0;
    }
  }
}
