// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest.testsrc.complexcase;

import com.google.devtools.build.android.desugar.nest.testsrc.complexcase.Bravo.Charlie;

/** Test class as source data. */
@SuppressWarnings({"PrivateConstructorForUtilityClass", "FieldCanBeFinal"}) // For testing.
public class Xylem {
  static class ConcreteAlpha implements Alpha {
    private ConcreteAlpha() {}

    private static long privateStaticField = ConcreteBravo.privateStaticMethod();
    private int privateInstanceField = 2;

    @Override
    public long abstractMethod(long x, int y) {
      return x + y + 1_000_000_000L;
    }
  }

  static class ConcreteBravo extends ConcreteAlpha implements Bravo {
    private ConcreteBravo() {}

    private static long privateStaticMethod() {
      return 1L;
    }

    private int privateInstanceMethod() {
      return super.privateInstanceField;
    }

    @Override
    public long abstractMethod(long x, int y) {
      return x + y + 2_000_000_000L;
    }
  }

  private static class XylemInvoker {
    private static long execute(long x, int y) {
      ConcreteBravo bravo = new ConcreteBravo();
      long localSum = x + y;
      Charlie charlie = (a, b) -> a + b + localSum;
      return ConcreteAlpha.privateStaticField
          + bravo.privateInstanceMethod()
          + Alpha.publicStaticMethod(new ConcreteAlpha(), bravo, x, y)
          + charlie.invokeCrossMatePrivateInstanceMethod();
    }
  }

  public static long execute(long x, int y) {
    return XylemInvoker.execute(x, y);
  }

  public static void main(String[] args) {
    int x = Integer.parseInt(args[0]);
    int y = Integer.parseInt(args[1]);
    System.out.println(execute(x, y));
  }
}
