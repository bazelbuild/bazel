package com.google.devtools.build.lib.rules.cpp;

import java.io.IOException;
import java.nio.channels.WritableByteChannel;
import java.util.Map;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import java.nio.file.Path;
import java.nio.file.Paths;

import static java.lang.Math.max;

public final class ClangHeaderMap {

  /** Logical representation of a bucket.
   The actual data is stored in the string pool.
   */
  private class HeaderMapBucket {
    private final String key;
    private final String prefix;
    private final String suffix;

    HeaderMapBucket(String key, String prefix, String suffix) {
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

  private static final int INT_SIZE = Integer.SIZE/8;

  /**
   Data stored in accordance to Clang's lexer types:

   <pre>{@code
  enum {
      HMAP_HeaderMagicNumber = ('h' << 24) | ('m' << 16) | ('a' << 8) | 'p',
      HMAP_HeaderVersion = 1,
      HMAP_EmptyBucketKey = 0
    };

    struct HeaderMapBucket {
      uint32_t Key;    // Offset (into strings) of key.
      uint32_t Prefix; // Offset (into strings) of value prefix.
      uint32_t Suffix; // Offset (into strings) of value suffix.
    };

    struct HMapHeader {
      uint32_t Magic;      // Magic word, also indicates byte order.
      uint16_t Version;      // Version number -- currently 1.
      uint16_t Reserved;     // Reserved for future use - zero for now.
      uint32_t StringsOffset;  // Offset to start of string pool.
      uint32_t NumEntries;     // Number of entries in the string table.
      uint32_t NumBuckets;     // Number of buckets (always a power of 2).
      uint32_t MaxValueLength; // Length of longest result path (excluding nul).
      // An array of 'NumBuckets' HeaderMapBucket objects follows this header.
      // Strings follow the buckets, at StringsOffset.
    };
  }</pre>
   */
  private ByteBuffer buffer;

  private int numBuckets;
  private int numUsedBuckets;
  private int stringsOffset;
  private int stringsSize;
  private int maxValueLength;
  private int maxStringsSize;

  // Used only for creation
  private HeaderMapBucket[] buckets;

  /**
  Create a header map from a raw Map of keys to strings
   Usage:
   A given path to a header is keyed by that header.
   i.e. Header.h -> Path/To/Header.h

   Additionally, it's possible to alias custom paths to headers.
   For example, it's possible to namespace a given target
   i.e. MyTarget/Header.h -> Path/To/Header.h

   The HeaderMap format is defined by the lexer of Clang
   https://clang.llvm.org/doxygen/HeaderMap_8cpp_source.html
   */
  ClangHeaderMap(Map<String, String> headerPathsByKeys) {
    int dataOffset = 1;
    setMap(headerPathsByKeys);

    int endBuckets = HEADER_SIZE + numBuckets * BUCKET_SIZE;
    stringsOffset = endBuckets - dataOffset;
    int totalBufferSize = endBuckets + maxStringsSize;
    buffer = ByteBuffer.wrap(new byte[totalBufferSize]).order(ByteOrder.LITTLE_ENDIAN);

    // Write out the header
    buffer.putInt(HEADER_MAGIC);
    buffer.putShort(HEADER_VERSION);
    buffer.putShort(HEADER_RESERVED);
    buffer.putInt(stringsOffset);

    // For each entry, we write a key, suffix, and prefix
    int stringPoolSize = headerPathsByKeys.size() * 3;
    buffer.putInt(stringPoolSize);
    buffer.putInt(numBuckets);
    buffer.putInt(maxValueLength);

    // Write out buckets and compute string offsets
    byte[] stringBytes = new byte[maxStringsSize];

    // Used to compute the current offset
    stringsSize = 0;
    for (int i = 0; i < numBuckets; i++) {
      HeaderMapBucket bucket = buckets[i];
      if (bucket == null) {
        buffer.putInt(EMPTY_BUCKET_KEY);
        buffer.putInt(0);
        buffer.putInt(0);
      } else {
        int keyOffset = stringsSize;
        buffer.putInt(keyOffset + dataOffset);
        stringsSize = addString(bucket.key, stringsSize, stringBytes);

        int prefixOffset = stringsSize;
        stringsSize = addString(bucket.prefix, stringsSize, stringBytes);
        buffer.putInt(prefixOffset + dataOffset);

        int suffixOffset = stringsSize;
        stringsSize = addString(bucket.suffix, stringsSize, stringBytes);
        buffer.putInt(suffixOffset + dataOffset);
      }
    }
    buffer.put(stringBytes, 0, stringsSize);
  }

  /*
  Write header map to a channel
   */
  public void writeToChannel(WritableByteChannel channel) throws IOException {
    buffer.flip();
    channel.write(buffer);
  }

  /**
   For testing purposes. Implement a similar algorithm as the clang lexer.
   */
  public String get(String key) {
    int bucketIdx = clangKeyHash(key) & (numBuckets - 1);
    while (bucketIdx < numBuckets) {
      // Buckets are right after the header
      int bucketOffset = HEADER_SIZE + (BUCKET_SIZE * bucketIdx);
      int keyOffset = buffer.getInt(bucketOffset);

      // Note: the lexer does a case insensitive compare here but
      // it isn't necessary for test purposes
      if (!key.equals(getString(keyOffset))) {
        bucketIdx++;
        continue;
      }

      // Start reading bytes from the prefix
      int prefixOffset = buffer.getInt(bucketOffset + INT_SIZE);
      int suffixOffset = buffer.getInt(bucketOffset + INT_SIZE * 2);
      return getString(prefixOffset) + getString(suffixOffset);
    }
    return null;
  }

  // Return a string from an offset
  // This method is used for testing only.
  private String getString(int offset) {
    int readOffset = stringsOffset + offset;
    int endStringsOffset = stringsOffset + stringsSize;
    int idx = readOffset;
    byte[] stringBytes = new byte[2048];
    while(idx < endStringsOffset) {
      byte c = (byte) buffer.getChar(idx);
      if (c == 0) {
        break;
      }
      stringBytes[idx] = c;
      idx++;
    }
    try {
      return new String(stringBytes).trim();
    } catch(Exception e) {
      return null;
    }
  }

  private void addBucket(HeaderMapBucket bucket, HeaderMapBucket[] buckets, int numBuckets) {
    // Use a load factor of 0.5
    if (((numUsedBuckets + 1) / numBuckets) > 0.5 == false) {
      int bucketIdx = clangKeyHash(bucket.key) & (numBuckets - 1);
      // Base case, the bucket Idx is free
      if (buckets[bucketIdx] == null) {
        buckets[bucketIdx] = bucket;
        this.numUsedBuckets++;
        return;
      }

      // Handle collisions.
      //
      // The lexer does a linear scan of the hash table when keys do
      // not match, starting at the bucket.
      while(bucketIdx < numBuckets) {
        bucketIdx = (bucketIdx + 1) & (numBuckets - 1);
        if (buckets[bucketIdx] == null) {
          buckets[bucketIdx] = bucket;
          this.numUsedBuckets++;
          return;
        }
      }
    }

    // If there are no more slots left, grow by a power of 2
    int newNumBuckets = numBuckets * 2;
    HeaderMapBucket[] newBuckets = new HeaderMapBucket[newNumBuckets];

    HeaderMapBucket[] oldBuckets = buckets;
    this.buckets = newBuckets;
    this.numBuckets = newNumBuckets;
    this.numUsedBuckets = 0;
    for(HeaderMapBucket cpBucket: oldBuckets) {
      if (cpBucket != null) {
        addBucket(cpBucket, newBuckets, newNumBuckets);
      }
    }

    // Start again
    addBucket(bucket, newBuckets, newNumBuckets);
  }

  private void setMap(Map<String, String> headerPathsByKeys){
    // Compute header metadata
    maxValueLength = 1;
    maxStringsSize = 0;
    numUsedBuckets = 0;

    // Per the format, buckets need to be powers of 2 in size
    numBuckets = getNextPowerOf2(headerPathsByKeys.size() + 1);
    buckets = new HeaderMapBucket[numBuckets];

    for(Map.Entry<String, String> entry: headerPathsByKeys.entrySet()){
      String key = entry.getKey();
      String path = entry.getValue();

      // Get the prefix and suffix
      String suffix;
      String prefix;
      Path pathValue = Paths.get(path);
      if (pathValue.getNameCount() < 2) {
        // The suffix is empty when the file path just a filename
        prefix = "";
        suffix = pathValue.getFileName().toString();
      } else {
        prefix = pathValue.getParent().toString() + "/";
        suffix = pathValue.getFileName().toString();
      }

      HeaderMapBucket bucket = new HeaderMapBucket(key, prefix, suffix);
      addBucket(bucket, buckets, numBuckets);
      int prefixLen = prefix.getBytes().length + 1;
      int suffixLen = suffix.getBytes().length + 1;
      int keyLen = key.getBytes().length + 1;
      maxStringsSize += prefixLen + suffixLen + keyLen;

      maxValueLength = max(maxValueLength, keyLen);
      maxValueLength = max(maxValueLength, suffixLen);
      maxValueLength = max(maxValueLength, prefixLen);
    }
  }

  // Utils
  private static int addString(String str, int totalLength, byte[] stringBytes) {
    for (byte b : str.getBytes(StandardCharsets.UTF_8)) {
      stringBytes[totalLength] = b;
      totalLength++;
    }
    stringBytes[totalLength] = (byte) 0;
    totalLength++;
    return totalLength;
  }

  private static int getNextPowerOf2(int a) {
    int b = 1;
    while (b < a) {
      b = b << 1;
    }
    return b;
  }

  // The same hashing algorithm as the Lexer.
  // Buckets must be inserted according to this.
  private static int clangKeyHash(String key) {
    // Keys are case insensitve.
    String lowerCaseKey = toLowerCaseAscii(key);
    int hash = 0;
    for (byte c : lowerCaseKey.getBytes(StandardCharsets.UTF_8)) {
      hash += c * 13;
    }
    return hash;
  }

  public static String toLowerCaseAscii(String string) {
    int length = string.length();
    StringBuilder builder = new StringBuilder(length);
    for (int i = 0; i < length; i++) {
      builder.append(toLowerCaseAscii(string.charAt(i)));
    }
    return builder.toString();
  }

  public static char toLowerCaseAscii(char c) {
    return isUpperCase(c) ? (char) (c ^ 0x20) : c;
  }

  public static boolean isUpperCase(char c) {
    return (c >= 'A') && (c <= 'Z');
  }
}
