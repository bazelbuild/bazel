// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.cpp;

import static java.lang.Math.ceil;
import static java.lang.Math.max;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;//?
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@Immutable
public final class CppHeaderMapAction extends AbstractFileWriteAction {

  private static final String GUID = "3f9eb099-e62d-4d3b-a08a-c478e188993d";

  // C++ header map of the current target
  private final CppHeaderMap cppHeaderMap;
  // Data required to build the actual header map
  // NOTE: If you add a field here, you'll likely need to add it to the cache key in computeKey().
  private final ImmutableList<CppHeaderMap> dependencies;
  private final String includePrefix = "";

  public CppHeaderMapAction(
      ActionOwner owner,
      CppHeaderMap cppHeaderMap,
      Iterable<CppHeaderMap> dependencies
  ) {
    super(
        owner,
        ImmutableList.<Artifact>builder()
        .addAll(Iterables.filter(privateHeaders, Artifact::isTreeArtifact))
        .addAll(Iterables.filter(publicHeaders, Artifact::isTreeArtifact))
        .build(),
        cppHeaderMap.getArtifact(),
        /*makeExecutable=*/ false);
    this.cppHeaderMap = cppHeaderMap;
    this.privateHeaders = ImmutableList.copyOf(privateHeaders);
    this.publicHeaders = ImmutableList.copyOf(publicHeaders);
    this.dependencies = ImmutableList.copyOf(dependencies);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext context) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        ByteBuffer buffer = serializeHeaderMap(headerMap);
        WritableByteChannel channel = Channels.newChannel(out);
        buffer.flip();
        channel.write(buffer);
        out.flush();
        out.close();
      }
    };
  }

  @Override
  public String getMnemonic() {
    return "CppHeaderMap";
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint f) {
    f.addString(GUID);
    for (Map.Entry<String, String> entry : headerMap.entrySet()) {
      String key = entry.getKey();
      String path = entry.getValue();
      f.addString(key + path);
    }
  }

  // Implementation of header map format follows

  private class ClangHeaderMapImpl {
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
                      addString(stringTable, addedStrings, parts[1])
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

      // Write header map header
      buffer.putInt(HEADER_MAGIC);
      buffer.putShort(HEADER_VERSION);
      buffer.putShort(HEADER_RESERVED);
      buffer.putInt(HEADER_SIZE + buckets.length * BUCKET_SIZE - dataOffset);
      buffer.putInt(numBuckets);
      buffer.putInt(buckets.length);
      buffer.putInt(maxValueLength);

      // write buckets
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

      // write string table
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

