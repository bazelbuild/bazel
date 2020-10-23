// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8.testdata.arithmetic;

import java.util.Arrays;

/** Test class */
@SuppressWarnings("PrivateConstructorForUtilityClass")
public class Arithmetic {
  static void addInts(int[] ints) {
    int result = 0;
    for (int i : ints) {
      result += i;
    }
    System.out.println("addInts: " + Arrays.toString(ints) + " = " + result);
  }

  static void addDoubles(double[] doubles) {
    double result = 0;
    for (double d : doubles) {
      result += d;
    }
    System.out.println("addDoubles: " + Arrays.toString(doubles) + " = " + result);
  }

  static void addLongs(long[] longs) {
    long result = 0;
    for (long l : longs) {
      result += l;
    }
    System.out.println("addLongs: " + Arrays.toString(longs) + " = " + result);
  }

  @SuppressWarnings("IdentityBinaryExpression")
  static void binaryOps() {
    int i = 0;
    System.out.println("i values:");
    i = i + 1;
    System.out.println(i);
    i = 1 + i;
    System.out.println(i);
    i = i * 4;
    System.out.println(i);
    i = i * i;
    System.out.println(i);
    i = 4 * i;
    System.out.println(i);
    i = i / 4;
    System.out.println(i);
    i = i / i;
    System.out.println(i);
    i = i % i;
    System.out.println(i);

    long l = 0;
    System.out.println("l values:");
    l = l + 1;
    System.out.println(l);
    l = 1 + l;
    System.out.println(l);
    l = l * 4;
    System.out.println(l);
    l = l * l;
    System.out.println(l);
    l = 4 * l;
    System.out.println(l);
    l = l / 4;
    System.out.println(l);
    l = l / l;
    System.out.println(l);
    l = l % l;
    System.out.println(l);

    double d = 0.0;
    System.out.println("d values: ");
    d = d + 1.0;
    System.out.println(d);
    d = 1.0 + d;
    System.out.println(d);
    d = d * 4.0;
    System.out.println(d);
    d = d * d;
    System.out.println(d);
    d = 4.0 * d;
    System.out.println(d);
    d = d / 4.0;
    System.out.println(d);
    d = d / d;
    System.out.println(d);
    d = d % d;
    System.out.println(d);

    float f = 0.0f;
    System.out.println("f values: ");
    f = f + 1.0f;
    System.out.println(f);
    f = 1.0f + f;
    System.out.println(f);
    f = f * 4.0f;
    System.out.println(f);
    f = f * f;
    System.out.println(f);
    f = 4.0f * f;
    System.out.println(f);
    f = f / 4.0f;
    System.out.println(f);
    f = f / f;
    System.out.println(f);
    f = f % f;
    System.out.println(f);
  }

  public static void moreOps() {
    int a = 42;
    int b = -a;
    int shiftLeftA = a << 5;
    int shiftRightA = a >> 5;
    int uShiftRightA = -a >>> 5;
    System.out.println(a + b + shiftLeftA + shiftRightA + uShiftRightA);
    float c = 42.42f;
    float d = -c;
    System.out.println(c + d);
    double e = 43.43;
    double f = -e;
    System.out.println(e + f);
    long g = 5000000000L;
    long h = -g;
    long shiftLeftG = g << 8;
    long shiftRightG = g >> 8;
    long uShiftRightG = -g >>> 8;
    System.out.println(g + h + shiftLeftG + shiftRightG + uShiftRightG);
  }

  public static void bitwiseInts(int x, int y) {
    System.out.println(x & y);
    System.out.println(x | y);
    System.out.println(x ^ y);
    System.out.println(~x);
  }

  public static void bitwiseLongs(long x, long y) {
    System.out.println(x & y);
    System.out.println(x | y);
    System.out.println(x ^ y);
    System.out.println(~x);
  }

  public static void main(String[] args) {
    addInts(new int[] {});
    addInts(new int[] {1});
    addInts(new int[] {0, 1, 2, 3});
    addDoubles(new double[] {0.0});
    addDoubles(new double[] {0.0, 1.0, 2.0});
    addDoubles(new double[] {0.0, 1.0, 2.0, 3.0});
    long l = 0x0000000100000000L;
    addLongs(new long[] {});
    addLongs(new long[] {l});
    addLongs(new long[] {l, l + 1, l + 2});
    binaryOps();
    moreOps();
    bitwiseInts(12345, 54321);
    bitwiseLongs(54321, 12345);
  }
}
