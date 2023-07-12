/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.stringconcat;

/** Test cases for string concatenations. */
public final class StringConcatTestCases {

  private static final String TEXT_CONSTANT = "<constant>";

  public static String simplePrefix(String content) {
    return "prefix:" + content;
  }

  public static String twoConcat(String x, String y) {
    return x + y;
  }

  public static <T> String twoConcat(String x, T y) {
    return "T:" + x + y;
  }

  public static String threeConcat(String x, String y, String z) {
    return x + y + z;
  }

  public static String twoConcatWithConstants(String x, String y) {
    return x + TEXT_CONSTANT + y;
  }

  public static String twoConcatWithRecipe(String x, String y) {
    return "<p>" + x + "<br>" + y + "</p>";
  }

  public static String twoConcatWithPrimitives(String x, int y) {
    return x + y;
  }

  public static String twoConcatWithPrimitives(String x, long y) {
    return x + y;
  }

  public static String twoConcatWithPrimitives(String x, double y) {
    return x + y;
  }

  public static String twoConcatWithPrimitives(int x, String y) {
    return x + y;
  }

  public static String twoConcatWithPrimitives(long x, String y) {
    return x + y;
  }

  public static String twoConcatWithPrimitives(double x, String y) {
    return x + y;
  }

  public static String threeConcatWithRecipe(String x, String y, String z) {
    return "<p>" + x + "<br>" + y + "<br>" + z + "</p>";
  }

  public static String concatWithAllPrimitiveTypes(
      String stringValue,
      int intValue,
      boolean booleanValue,
      byte byteValue,
      char charValue,
      short shortValue,
      double doubleValue,
      float floatValue,
      long longValue) {
    return stringValue
        + '/'
        + intValue
        + '/'
        + booleanValue
        + '/'
        + byteValue
        + '/'
        + charValue
        + '/'
        + shortValue
        + '/'
        + doubleValue
        + '/'
        + floatValue
        + '/'
        + longValue;
  }

  private StringConcatTestCases() {}
}
