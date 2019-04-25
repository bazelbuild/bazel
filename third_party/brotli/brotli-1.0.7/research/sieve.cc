#include "./sieve.h"

/* Pointer to position in (combined corpus) text. */
typedef uint32_t TextIdx;

/* Index of sample / generation. */
typedef uint16_t SampleIdx;

typedef struct Slot {
  TextIdx next;
  TextIdx offset;
  SampleIdx presence;
  SampleIdx mark;
} Slot;

static const TextIdx kNowhere = static_cast<TextIdx>(-1);

static TextIdx dryRun(TextIdx sliceLen, Slot* map, TextIdx* shortcut,
    TextIdx end, TextIdx middle, SampleIdx minPresence, SampleIdx iteration) {
  TextIdx from = kNowhere;
  TextIdx to = kNowhere;
  TextIdx result = 0;
  SampleIdx targetPresence = minPresence;
  for (TextIdx i = 0; i < end; ++i) {
    if (i == middle) {
      targetPresence++;
    }
    Slot& item = map[shortcut[i]];
    if (item.mark != iteration) {
      item.mark = iteration;
      if (item.presence >= targetPresence) {
        if ((to == kNowhere) || (to < i)) {
          if (from != kNowhere) {
            result += to - from;
          }
          from = i;
        }
        to = i + sliceLen;
      }
    }
  }
  if (from != kNowhere) {
    result += to - from;
  }
  return result;
}

static std::string createDictionary(const uint8_t* data, TextIdx sliceLen,
    Slot* map, TextIdx* shortcut, TextIdx end, TextIdx middle,
    SampleIdx minPresence, SampleIdx iteration) {
  std::string output;
  TextIdx from = kNowhere;
  TextIdx to = kNowhere;
  SampleIdx targetPresence = minPresence;
  for (TextIdx i = 0; i < end; ++i) {
    if (i == middle) {
      targetPresence++;
    }
    Slot& item = map[shortcut[i]];
    if (item.mark != iteration) {
      item.mark = iteration;
      if (item.presence >= targetPresence) {
        if ((to == kNowhere) || (to < i)) {
          if (from != kNowhere) {
            output.insert(output.end(), &data[from], &data[to]);
          }
          from = i;
        }
        to = i + sliceLen;
      }
    }
  }
  if (from != kNowhere) {
    output.insert(output.end(), &data[from], &data[to]);
  }
  return output;
}

std::string sieve_generate(size_t dictionary_size_limit, size_t slice_len,
    const std::vector<size_t>& sample_sizes, const uint8_t* sample_data) {
  /* Parameters aliasing */
  TextIdx targetSize = static_cast<TextIdx>(dictionary_size_limit);
  if (targetSize != dictionary_size_limit) {
    fprintf(stderr, "dictionary_size_limit is too large\n");
    return "";
  }
  TextIdx sliceLen = static_cast<TextIdx>(slice_len);
  if (sliceLen != slice_len) {
    fprintf(stderr, "slice_len is too large\n");
    return "";
  }
  if (sliceLen < 1) {
    fprintf(stderr, "slice_len is too small\n");
    return "";
  }
  SampleIdx numSamples = static_cast<SampleIdx>(sample_sizes.size());
  if ((numSamples != sample_sizes.size()) || (numSamples * 2 < numSamples)) {
    fprintf(stderr, "too many samples\n");
    return "";
  }
  const uint8_t* data = sample_data;

  TextIdx total = 0;
  std::vector<TextIdx> offsets;
  for (SampleIdx i = 0; i < numSamples; ++i) {
    TextIdx delta = static_cast<TextIdx>(sample_sizes[i]);
    if (delta != sample_sizes[i]) {
      fprintf(stderr, "sample is too large\n");
      return "";
    }
    if (delta == 0) {
      fprintf(stderr, "empty samples are prohibited\n");
      return "";
    }
    if (total + delta <= total) {
      fprintf(stderr, "corpus is too large\n");
      return "";
    }
    total += delta;
    offsets.push_back(total);
  }

  if (total * 2 < total) {
    fprintf(stderr, "corpus is too large\n");
    return "";
  }

  if (total < sliceLen) {
    fprintf(stderr, "slice_len is larger than corpus size\n");
    return "";
  }

  /*****************************************************************************
   * Build coverage map.
   ****************************************************************************/
  std::vector<Slot> map;
  std::vector<TextIdx> shortcut;
  map.push_back({0, 0, 0, 0});
  TextIdx end = total - sliceLen;
  TextIdx hashLen = 11;
  while (hashLen < 29 && ((1u << hashLen) < end)) {
    hashLen += 3;
  }
  hashLen -= 3;
  TextIdx hashMask = (1u << hashLen) - 1u;
  std::vector<TextIdx> hashHead(1 << hashLen);
  TextIdx hashSlot = 1;
  SampleIdx piece = 0;
  TextIdx hash = 0;
  TextIdx lShift = 3;
  TextIdx rShift = hashLen - lShift;
  for (TextIdx i = 0; i < sliceLen - 1; ++i) {
    TextIdx v = data[i];
    hash = (((hash << lShift) | (hash >> rShift)) & hashMask) ^ v;
  }
  TextIdx lShiftX = (lShift * (sliceLen - 1)) % hashLen;
  TextIdx rShiftX = hashLen - lShiftX;
  for (TextIdx i = 0; i < end; ++i) {
    TextIdx v = data[i + sliceLen - 1];
    hash = (((hash << lShift) | (hash >> rShift)) & hashMask) ^ v;

    if (offsets[piece] == i) {
      piece++;
    }
    TextIdx slot = hashHead[hash];
    while (slot != 0) {
      Slot& item = map[slot];
      TextIdx start = item.offset;
      bool miss = false;
      for (TextIdx j = 0; j < sliceLen; ++j) {
        if (data[i + j] != data[start + j]) {
          miss = true;
          break;
        }
      }
      if (!miss) {
        if (item.mark != piece) {
          item.mark = piece;
          item.presence++;
        }
        shortcut.push_back(slot);
        break;
      }
      slot = item.next;
    }
    if (slot == 0) {
      map.push_back({hashHead[hash], i, 1, piece});
      hashHead[hash] = hashSlot;
      shortcut.push_back(hashSlot);
      hashSlot++;
    }
    v = data[i];
    hash ^= ((v << lShiftX) | (v >> rShiftX)) & hashMask;
  }

  /*****************************************************************************
   * Build dictionary of specified size.
   ****************************************************************************/
  SampleIdx a = 1;
  TextIdx size = dryRun(
      sliceLen, map.data(), shortcut.data(), end, end, a, ++piece);
  /* Maximal output is smaller than target. */
  if (size <= targetSize) {
    return createDictionary(
        data, sliceLen, map.data(), shortcut.data(), end, end, a, ++piece);
  }

  SampleIdx b = numSamples;
  size = dryRun(sliceLen, map.data(), shortcut.data(), end, end, b, ++piece);
  if (size == targetSize) {
    return createDictionary(
        data, sliceLen, map.data(), shortcut.data(), end, end, b, ++piece);
  }
  /* Run binary search. */
  if (size < targetSize) {
    /* size(a) > targetSize > size(b) && a < m < b */
    while (a + 1 < b) {
      SampleIdx m = static_cast<SampleIdx>((a + b) / 2);
      size = dryRun(
          sliceLen, map.data(), shortcut.data(), end, end, m, ++piece);
      if (size < targetSize) {
        b = m;
      } else if (size > targetSize) {
        a = m;
      } else {
        return createDictionary(
            data, sliceLen, map.data(), shortcut.data(), end, end, b, ++piece);
      }
    }
  } else {
    a = b;
  }
  /* size(minPresence) > targetSize > size(minPresence + 1) */
  SampleIdx minPresence = a;
  TextIdx c = 0;
  TextIdx d = end;
  /* size(a) < targetSize < size(b) && a < m < b */
  while (c + 1 < d) {
    TextIdx m = (c + d) / 2;
    size = dryRun(
        sliceLen, map.data(), shortcut.data(), end, m, minPresence, ++piece);
    if (size < targetSize) {
      c = m;
    } else if (size > targetSize) {
      d = m;
    } else {
      return createDictionary(data, sliceLen, map.data(), shortcut.data(), end,
          m, minPresence, ++piece);
    }
  }

  bool unrestricted = false;
  if (minPresence <= 2 && !unrestricted) {
    minPresence = 2;
    c = end;
  }
  return createDictionary(data, sliceLen, map.data(), shortcut.data(), end, c,
      minPresence, ++piece);
}
