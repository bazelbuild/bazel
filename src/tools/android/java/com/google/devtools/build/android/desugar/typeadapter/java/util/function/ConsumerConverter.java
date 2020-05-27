/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.typeadapter.java.util.function;

/**
 * Converts types between the desugar-mirrored and desugar-shadowed {@link
 * java.util.function.Consumer}.
 */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class ConsumerConverter {

  private ConsumerConverter() {}

  public static <T> j$.util.function.Consumer<T> from(
      final java.util.function.Consumer<T> consumer) {
    return consumer == null
        ? null
        : new j$.util.function.Consumer<T>() {
          @Override
          public void accept(T t) {
            consumer.accept(t);
          }

          @Override
          public j$.util.function.Consumer<T> andThen(j$.util.function.Consumer<? super T> after) {
            return from(consumer.andThen(to(after)));
          }
        };
  }

  public static <T> java.util.function.Consumer<T> to(final j$.util.function.Consumer<T> consumer) {
    return consumer == null
        ? null
        : new java.util.function.Consumer<T>() {
          @Override
          public void accept(T t) {
            consumer.accept(t);
          }
        };
  }
}
