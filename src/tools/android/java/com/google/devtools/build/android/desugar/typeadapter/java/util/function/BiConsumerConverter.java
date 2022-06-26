/*
 * Copyright 2021 The Bazel Authors. All rights reserved.
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
 * java.util.function.BiConsumer}.
 */
@SuppressWarnings({"AndroidJdkLibsChecker", "UnnecessarilyFullyQualified", "NewApi"})
public abstract class BiConsumerConverter {

  private BiConsumerConverter() {}

  public static <T, U> j$.util.function.BiConsumer<T, U> from(
      final java.util.function.BiConsumer<T, U> biConsumer) {
    return biConsumer == null
        ? null
        : new j$.util.function.BiConsumer<T, U>() {
          @Override
          public void accept(T t, U u) {
            biConsumer.accept(t, u);
          }

          @Override
          public j$.util.function.BiConsumer<T, U> andThen(
              j$.util.function.BiConsumer<? super T, ? super U> after) {
            return from(biConsumer.andThen(to(after)));
          }
        };
  }

  public static <T, U> java.util.function.BiConsumer<T, U> to(
      final j$.util.function.BiConsumer<T, U> biConsumer) {
    return biConsumer == null
        ? null
        : new java.util.function.BiConsumer<T, U>() {
          @Override
          public void accept(T t, U u) {
            biConsumer.accept(t, u);
          }
        };
  }
}
