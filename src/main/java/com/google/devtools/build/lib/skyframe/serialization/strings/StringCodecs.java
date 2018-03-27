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

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;

/** Utility for accessing (potentially platform-specific) {@link String} {@link ObjectCodec}s. */
public final class StringCodecs {

  private static final StringCodec stringCodec = new StringCodec();

  private StringCodecs() {}

  /**
   * Returns whether or not optimized codecs are available. Exposed so users can check at runtime
   * if the expected optimizations are applied.
   */
  public static boolean supportsOptimizedAscii() {
    return false;
  }

  /**
   * Returns singleton instance optimized for almost-always ASCII data, if supported. Otherwise,
   * returns a functional, but not optimized implementation. To tell if the optimized version is
   * supported see {@link #supportsOptimizedAscii()}.
   *
   * <p>Note that when optimized, this instance can still serialize/deserialize UTF-8 data, but with
   *  potentially worse performance than {@link #simple()}.
   *
   * <p>Currently this is the same as {@link #simple()}, it remains to avoid a time-consuming
   * cleanup and in case we want to revive an optimized version in the near future.
   */
  // TODO(bazel-core): Determine if we need to revive ascii-optimized.
  public static ObjectCodec<String> asciiOptimized() {
    return simple();
  }

  /**
   * Returns singleton instance of basic implementation. Should be preferred over
   * {@link #asciiOptimized()} when a sufficient amount of UTF-8 data is expected.
   */
  public static ObjectCodec<String> simple() {
    return stringCodec;
  }
}
