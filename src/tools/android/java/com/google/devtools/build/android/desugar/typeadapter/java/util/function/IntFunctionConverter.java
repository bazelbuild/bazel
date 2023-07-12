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
 * java.util.function.IntFunction}.
 */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class IntFunctionConverter {

  private IntFunctionConverter() {}

  public static <R> j$.util.function.IntFunction<R> from(
      final java.util.function.IntFunction<R> function) {
    return function == null
        ? null
        : new j$.util.function.IntFunction<R>() {
          @Override
          public R apply(int value) {
            return function.apply(value);
          }
        };
  }

  public static <R> java.util.function.IntFunction<R> to(
      final j$.util.function.IntFunction<R> function) {
    return function == null
        ? null
        : new java.util.function.IntFunction<R>() {
          @Override
          public R apply(int value) {
            return function.apply(value);
          }
        };
  }
}
