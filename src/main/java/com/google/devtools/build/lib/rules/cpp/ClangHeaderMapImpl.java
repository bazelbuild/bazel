package com.google.devtools.build.lib.rules.cpp;

import static java.lang.Math.ceil;
import static java.lang.Math.max;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class ClangHeaderMapImpl {
  // Clang's Header map implementation

  /**
   * Logical representation of a hash bucket. The actual data is stored in the string pool.
   */
  private class Bucket {

    private final int key;
    private final int prefix;
    private final int suffix;

    Bucket(int key, int prefix, int suffix) {
      this.key = key;
      this.prefix = prefix;
      this.suffix = suffix;
    }
  }

  private static final int HEADER_MAGIC = ('h' << 24) | ('m' << 16) | ('a' << 8) | 'p';
  private static final short HEADER_VERSION = 1;
  private static final short HEADER_RESERVED = 0;
  private static final int EMPTY_BUCKET_KEY = 0;

  private static final int HEADER_SIZE = 24;
  private static final int BUCKET_SIZE = 12;

  private static final int DEFAULT_NUM_BUCKETS = 256;
  private static final double MAX_LOAD_FACTOR = 0.5;


  private int numBuckets;
  private int maxValueLength;

  private final Bucket[] buckets;
  private final byte[] stringBytes;

  /**
   * Create a header map buffer from a map of keys to strings Usage: A given path to a header is
   * keyed by that header. i.e. Header.h -> Path/To/Header.h
   *
   * Additionally, it's possible to alias custom paths to headers. For example, it's possible to
   * namespace a given target i.e. MyTarget/Header.h -> Path/To/Header.h
   *
   * The HeaderMap format is defined by the lexer of Clang https://clang.llvm.org/doxygen/HeaderMap_8cpp_source.html
   */
  ClangHeaderMapImpl(Map<String, Path> entries) {
    Map<String, Bucket> bucketMap = new HashMap<>();
    Map<String, Integer> addedStrings = new HashMap<>();
    ByteArrayOutputStream stringTable = new ByteArrayOutputStream();

    // Pre-compute the buckets to de-duplicate paths
    entries.forEach(
        (key, path) -> {
          bucketMap.computeIfAbsent(
              Ascii.toLowerCase(key),
              lowercaseKey -> {
                String[] parts = splitPath(path);
                maxValueLength = max(maxValueLength, parts[0].length() + parts[1].length());
                return new Bucket(
                    addString(stringTable, addedStrings, key),
                    addString(stringTable, addedStrings, parts[0]),
                    addString(stringTable, addedStrings, parts[1]))
                );
              });
        });

    long numBucketsLong = max(DEFAULT_NUM_BUCKETS,
        nextPowerOf2((long) ceil(entries.size() / MAX_LOAD_FACTOR)));
    Preconditions.checkState(Integer.MAX_VALUE > numBucketsLong,
        "Clang Header Maps overflowed number of buckets");
    int numBuckets = (int) numBucketsLong;

    Bucket[] buckets = new Bucket[numBuckets];
    bucketMap.forEach(
        (key, bucket) -> {
          final int hash0 = clangHash(key) & (numBuckets - 1);
          int hash = hash0;
          do {
            if (buckets[hash] == null) {
              // spot is free, insert a new bucket
              buckets[hash] = bucket;
              return;
            }
            // linear probing for collisions
            hash = (hash + 1) & (numBuckets - 1);
          } while (hash != hash0);
          Preconditions.checkState(false,
              "Clang Header Maps overflowed bucket table but we allocated enough buckets.");
        });
    this.buckets = buckets;
    this.stringBytes= stringTable.toByteArray();
    this.numBuckets = bucketMap.size();
  }

  @VisibleForTesting
  private static String[] splitPath(Path path) {
    String[] result = new String[2];
    if (path.getNameCount() < 2) {
      result[0] = "";
      result[1] = path.toString();
    } else {
      result[0] = path.getParent() + "/";
      result[1] = path.getFileName().toString();
    }
    return result;
  }

  private static int addString(ByteArrayOutputStream stringTable, Map<String, Integer> addedStrings, String str) {
    Integer seenOffset = addedStrings.get(str);
    if (seenOffset != null) {
      // we've already stored this string in the stringTable, reuse it
      return seenOffset;
    }
    int offset = stringTable.size();
    try {
      stringTable.write(str.getBytes(StandardCharsets.UTF_8));
    } catch (IOException e) {
      throw new IllegalStateException("ByteArrayOutputStream caused IOException");
    }
    stringTable.write(0);
    addedStrings.put(str, offset);
    return offset;
  }

  public void serialize(ByteBuffer buffer) {
    int dataOffset = 1;
    buffer.putInt(HEADER_MAGIC);
    buffer.putShort(HEADER_VERSION);
    buffer.putShort(HEADER_RESERVED);
    buffer.putInt(HEADER_SIZE + buckets.length * BUCKET_SIZE - dataOffset);
    buffer.putInt(numBuckets);
    buffer.putInt(buckets.length);
    buffer.putInt(maxValueLength);

    for (Bucket bucket: buckets) {
      if (bucket == null) {
        buffer.putInt(EMPTY_BUCKET_KEY);
        buffer.putInt(0);
        buffer.putInt(0);
      } else {
        buffer.putInt(bucket.key + dataOffset);
        buffer.putInt(bucket.prefix + dataOffset);
        buffer.putInt(bucket.suffix + dataOffset);
      }
    }
    buffer.put(stringBytes);
  }

  // Utility Functions
  private static long nextPowerOf2(long a) {
    return (a & (a - 1)) == 0
        ? a // power of 2
        : Long.highestOneBit(a) << 1; // next power of 2
  }

  // The same hashing algorithm as clang's Lexer.
  // Buckets must be inserted according to this.
  private static int clangHash(String key) {
    // Keys are case insensitive.
    int hash = 0;
    for (byte c : Ascii.toLowerCase(key).getBytes(StandardCharsets.UTF_8)) {
      hash += c * 13;
    }
    return hash;
  }
}

}
