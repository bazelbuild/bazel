/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.google.devtools.build.android.desugar.typeadapter.java.util.function;

/**
 * Converts types between the desugar-mirrored and desugar-shadowed {@link
 * java.util.function.IntConsumer}.
 */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class IntConsumerConverter {

  private IntConsumerConverter() {}

  public static j$.util.function.IntConsumer from(final java.util.function.IntConsumer consumer) {
    return consumer == null
        ? null
        : new j$.util.function.IntConsumer() {

          @Override
          public void accept(int value) {
            consumer.accept(value);
          }

          @Override
          public j$.util.function.IntConsumer andThen(j$.util.function.IntConsumer after) {
            return from(consumer.andThen(to(after)));
          }
        };
  }

  public static java.util.function.IntConsumer to(final j$.util.function.IntConsumer consumer) {
    return consumer == null
        ? null
        : new java.util.function.IntConsumer() {
          @Override
          public void accept(int value) {
            consumer.accept(value);
          }
        };
  }
}
