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
package com.google.devtools.build.android.desugar.testdata;

/** This class calls Long.compare(long, long) */
public class ClassCallingLongCompare {

  public static int compareLongByCallingLong_compare(long a, long b) {
    return Long.compare(a, b);
  }

  public static String compareLongByCallingLong_compare2(long a, long b) {
    if (Long.compare(a, b) == 0) {
      return "e";
    }
    if (Long.compare(a, b) > 0) {
      return "g";
    }
    if (Long.compare(a, b) < 0) {
      return "l";
    }
    throw new AssertionError("unreachable");
  }

  public static int compareLongWithLambda(long a, long b) {
    return internalCompare(a, b, (long l1, long l2) -> Long.compare(l1, l2));
  }

  public static int compareLongWithMethodReference(long a, long b) {
    return internalCompare(a, b, Long::compare);
  }

  private static interface LongCmpFunc {
    int compare(long a, long b);
  }

  private static int internalCompare(long a, long b, LongCmpFunc func) {
    return func.compare(a, b);
  }
}
