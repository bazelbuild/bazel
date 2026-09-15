// Copyright 2015 The Bazel Authors. All rights reserved.
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


import java.io.IOException;

public class B extends A {

  B() {
    // Will trigger compile error if Signature attr is missing.
    String str = protectedMethod("foo");

    try {
      deprecatedMethod(); // <-- triggers deprecation warning; checked by
                          // .sh script.
    } catch (IOException e) { // Will trigger compile error if Exceptions
                              // annotation is discarded.
    }

    l(A.L1); // <-- should be a compile-time ConstantValue; checked by .sh.

    new PublicInner();
  }

  @MyAnnotation
  void l(long l) {}

  @RuntimeAnnotation
  public int k;

  public static void main(String[] args) throws Exception {
    // Regression test for bug #1210750.
    if (!Class.forName("B").getField("k").isAnnotationPresent(RuntimeAnnotation.class)) {
      throw new AssertionError("RuntimeAnnotation got lost!");
    }

    System.err.println("B.main() OK");
  }

}
