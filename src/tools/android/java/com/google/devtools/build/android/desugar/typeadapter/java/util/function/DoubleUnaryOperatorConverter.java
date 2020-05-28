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

import j$.util.function.DoubleUnaryOperator;

/**
 * Converts types between the desugar-mirrored and desugar-shadowed {@link
 * java.util.function.DoubleUnaryOperator}.
 */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class DoubleUnaryOperatorConverter {

  private DoubleUnaryOperatorConverter() {}

  public static j$.util.function.DoubleUnaryOperator from(
      final java.util.function.DoubleUnaryOperator function) {
    return function == null
        ? null
        : new j$.util.function.DoubleUnaryOperator() {

          @Override
          public double applyAsDouble(double operand) {
            return function.applyAsDouble(operand);
          }

          @Override
          public DoubleUnaryOperator compose(DoubleUnaryOperator before) {
            return from(function.compose(to(before)));
          }

          @Override
          public DoubleUnaryOperator andThen(DoubleUnaryOperator after) {
            return from(function.andThen(to(after)));
          }
        };
  }

  public static java.util.function.DoubleUnaryOperator to(
      final j$.util.function.DoubleUnaryOperator function) {
    return function == null
        ? null
        : new java.util.function.DoubleUnaryOperator() {
          @Override
          public double applyAsDouble(double operand) {
            return function.applyAsDouble(operand);
          }
        };
  }
}
