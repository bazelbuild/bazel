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
 * java.util.function.BiFunction}.
 */
@SuppressWarnings({"AndroidJdkLibsChecker", "UnnecessarilyFullyQualified"})
public abstract class BiFunctionConverter {

  private BiFunctionConverter() {}

  public static <T, U, R> j$.util.function.BiFunction<T, U, R> from(
      final java.util.function.BiFunction<T, U, R> function) {
    return function == null
        ? null
        : new j$.util.function.BiFunction<T, U, R>() {
          @Override
          public R apply(T t, U u) {
            return function.apply(t, u);
          }

          @Override
          public <V> j$.util.function.BiFunction<T, U, V> andThen(
              j$.util.function.Function<? super R, ? extends V> after) {
            return from(function.andThen(FunctionConverter.to(after)));
          }
        };
  }

  public static <T, U, R> java.util.function.BiFunction<T, U, R> to(
      final j$.util.function.BiFunction<T, U, R> function) {
    return function == null
        ? null
        : new java.util.function.BiFunction<T, U, R>() {
          @Override
          public R apply(T t, U u) {
            return function.apply(t, u);
          }
        };
  }
}
