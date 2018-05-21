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

import static com.google.devtools.build.lib.skyframe.serialization.UnsafeJdk9StringCodec.canUseUnsafeCodec;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skyframe.serialization.CodecRegisterer;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.UnsafeJdk9StringCodec;

/** Utility for accessing (potentially platform-specific) {@link String} {@link ObjectCodec}s. */
public final class StringCodecs {

  private static final StringCodec stringCodec = new StringCodec();

  private static final UnsafeJdk9StringCodec unsafeCodec =
      canUseUnsafeCodec() ? new UnsafeJdk9StringCodec() : null;

  /**
   * Returns optimized singleton instance, if supported. Otherwise, returns a functional, but not
   * optimized implementation. Currently supported on JDK9.
   */
  public static ObjectCodec<String> asciiOptimized() {
    return unsafeCodec != null ? unsafeCodec : stringCodec;
  }

  static class UnsafeStringCodecRegisterer implements CodecRegisterer<UnsafeJdk9StringCodec> {
    @Override
    public Iterable<? extends ObjectCodec<?>> getCodecsToRegister() {
      return canUseUnsafeCodec() ? ImmutableList.of(unsafeCodec) : ImmutableList.of();
    }
  }

  static class SimpleStringCodecRegisterer implements CodecRegisterer<StringCodec> {
    @Override
    public Iterable<StringCodec> getCodecsToRegister() {
      return canUseUnsafeCodec() ? ImmutableList.of() : ImmutableList.of(stringCodec);
    }
  }

  /** Returns singleton instance of basic implementation. */
  public static ObjectCodec<String> simple() {
    return stringCodec;
  }
}
