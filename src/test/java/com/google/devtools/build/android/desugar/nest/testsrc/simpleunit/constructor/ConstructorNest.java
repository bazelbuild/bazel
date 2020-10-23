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

package com.google.devtools.build.android.desugar.nest.testsrc.simpleunit.constructor;

/** A nest for testing private constructor desugaring. */
public class ConstructorNest {

  /** A nest member class that encloses private constructors with cross-mate invocations. */
  public static class ConstructorServiceMate {
    private final long x;
    private final int y;

    private ConstructorServiceMate() throws Exception {
      this(10L, 20);
    }

    private ConstructorServiceMate(long x, int y) throws Exception {
      this.x = x;
      this.y = y;
    }

    long getSum() {
      return x + y;
    }
  }

  public static long createFromZeroArgConstructor() throws Exception {
    return new ConstructorServiceMate().getSum();
  }

  public static long createFromMultiArgConstructor(long x, int y) throws Exception {
    return new ConstructorServiceMate(x, y).getSum();
  }
}
