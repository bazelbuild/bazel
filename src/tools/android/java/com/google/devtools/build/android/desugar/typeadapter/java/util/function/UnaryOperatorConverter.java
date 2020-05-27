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
 * java.util.function.UnaryOperator}.
 */
@SuppressWarnings({"AndroidJdkLibsChecker", "UnnecessarilyFullyQualified"})
public abstract class UnaryOperatorConverter {

  private UnaryOperatorConverter() {}

  public static <T> j$.util.function.UnaryOperator<T> from(
      final java.util.function.UnaryOperator<T> unaryOperator) {
    return unaryOperator == null
        ? null
        : new j$.util.function.UnaryOperator<T>() {
          @Override
          public T apply(T t) {
            return unaryOperator.apply(t);
          }

          @Override
          public <V> j$.util.function.Function<V, T> compose(
              j$.util.function.Function<? super V, ? extends T> before) {
            return FunctionConverter.from(unaryOperator.compose(FunctionConverter.to(before)));
          }

          @Override
          public <V> j$.util.function.Function<T, V> andThen(
              j$.util.function.Function<? super T, ? extends V> after) {
            return FunctionConverter.from(unaryOperator.andThen(FunctionConverter.to(after)));
          }
        };
  }

  public static <T> java.util.function.UnaryOperator<T> to(
      final j$.util.function.UnaryOperator<T> unaryOperator) {
    return unaryOperator == null
        ? null
        : new java.util.function.UnaryOperator<T>() {
          @Override
          public T apply(T t) {
            return unaryOperator.apply(t);
          }
        };
  }
}
