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
 * java.util.function.Predicate}.
 */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class PredicateConverter {

  private PredicateConverter() {}

  public static <T> j$.util.function.Predicate<T> from(
      final java.util.function.Predicate<T> predicate) {
    return predicate == null
        ? null
        : new j$.util.function.Predicate<T>() {
          @Override
          public boolean test(T t) {
            return predicate.test(t);
          }

          @Override
          public j$.util.function.Predicate<T> and(j$.util.function.Predicate<? super T> other) {
            return from(predicate.and(to(other)));
          }

          @Override
          public j$.util.function.Predicate<T> negate() {
            return from(predicate.negate());
          }

          @Override
          public j$.util.function.Predicate<T> or(j$.util.function.Predicate<? super T> other) {
            return from(predicate.or(to(other)));
          }
        };
  }

  public static <T> java.util.function.Predicate<T> to(
      final j$.util.function.Predicate<T> predicate) {
    return predicate == null
        ? null
        : new java.util.function.Predicate<T>() {
          @Override
          public boolean test(T t) {
            return predicate.test(t);
          }
        };
  }
}
