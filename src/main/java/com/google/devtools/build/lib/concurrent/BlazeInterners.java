// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.common.collect.Interners.InternerBuilder;

/** Wrapper around {@link Interners}, with Blaze-specific predetermined concurrency levels. */
public class BlazeInterners {
  private static final int DEFAULT_CONCURRENCY_LEVEL = Runtime.getRuntime().availableProcessors();
  private static final int CONCURRENCY_LEVEL;

  static {
    String val = System.getenv("BLAZE_INTERNER_CONCURRENCY_LEVEL");
    CONCURRENCY_LEVEL = (val == null) ? DEFAULT_CONCURRENCY_LEVEL : Integer.parseInt(val);
  }

  private static InternerBuilder setConcurrencyLevel(InternerBuilder builder) {
    return builder.concurrencyLevel(CONCURRENCY_LEVEL);
  }

  public static <T> Interner<T> newWeakInterner() {
    return setConcurrencyLevel(Interners.newBuilder().weak()).build();
  }

  public static <T> Interner<T> newStrongInterner() {
    return setConcurrencyLevel(Interners.newBuilder().strong()).build();
  }
}

