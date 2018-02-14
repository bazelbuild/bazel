// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.strings;

import com.google.devtools.build.lib.skyframe.serialization.CodecRegisterer;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecRegistry;
import java.util.logging.Logger;

/** Utility for accessing (potentially platform-specific) {@link String} {@link ObjectCodec}s. */
public final class StringCodecs {

  private static final Logger logger = Logger.getLogger(StringCodecs.class.getName());

  private static final StringCodec stringCodec;
  private static final ObjectCodec<String> asciiOptimized;

  static {
    stringCodec = new StringCodec();
    if (FastStringCodec.isAvailable()) {
      asciiOptimized = new FastStringCodec();
    } else {
      logger.warning("Optimized string deserialization unavailable");
      asciiOptimized = stringCodec;
    }
  }

  private StringCodecs() {}

  /**
   * Returns whether or not optimized codecs are available. Exposed so users can check at runtime
   * if the expected optimizations are applied.
   */
  public static boolean supportsOptimizedAscii() {
    return asciiOptimized instanceof FastStringCodec;
  }

  /**
   * Returns singleton instance optimized for almost-always ASCII data, if supported. Otherwise,
   * returns a functional, but not optimized implementation. To tell if the optimized version is
   * supported see {@link #supportsOptimizedAscii()}.
   *
   * <p>Note that when optimized, this instance can still serialize/deserialize UTF-8 data, but with
   *  potentially worse performance than {@link #simple()}.
   */
  public static ObjectCodec<String> asciiOptimized() {
    return asciiOptimized;
  }

  /**
   * Returns singleton instance of basic implementation. Should be preferred over
   * {@link #asciiOptimized()} when a sufficient amount of UTF-8 data is expected.
   */
  public static ObjectCodec<String> simple() {
    return stringCodec;
  }

  /**
   * Registers a codec for {@link String}.
   *
   * <p>Needed to resolve ambiguity between {@link StringCodec} and {@link FastStringCodec}.
   */
  static class StringCodecRegisterer implements CodecRegisterer<StringCodec> {
    @Override
    public void register(ObjectCodecRegistry.Builder builder) {
      if (!supportsOptimizedAscii()) {
        builder.add(String.class, simple());
      }
    }
  }

  /**
   * Registers a codec for {@link String}.
   *
   * <p>Needed to resolve ambiguity between {@link StringCodec} and {@link FastStringCodec}.
   */
  static class FastStringCodecRegisterer implements CodecRegisterer<FastStringCodec> {
    @Override
    public void register(ObjectCodecRegistry.Builder builder) {
      if (supportsOptimizedAscii()) {
        builder.add(String.class, asciiOptimized());
      }
    }
  }
}
