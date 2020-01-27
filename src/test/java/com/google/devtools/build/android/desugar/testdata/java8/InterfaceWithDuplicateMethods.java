// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.testdata.java8;

/**
 * Test for b/38308515. This interface has one instance method {@code m()} and one static method
 * {@code m(InterfaceWithDuplicateMethods)}, which may cause Desugar to dump a companion class with
 * duplicate method signatures.
 */
public interface InterfaceWithDuplicateMethods {

  /**
   * In the companion class, this default method will be transformed to {@code int
   * getZero(InterfaceWithDuplicateMethods)}, which has the same signature as the static interface
   * method below.
   */
  @SuppressWarnings("AmbiguousMethodReference")
  default int getZero() {
    return 0;
  }

  /** Should not be called. Should only be called by the class {@link ClassWithDuplicateMethods} */
  @SuppressWarnings("AmbiguousMethodReference")
  static int getZero(InterfaceWithDuplicateMethods i) {
    return 1;
  }

  /** This class implements the interface, and calls the static interface method. */
  class ClassWithDuplicateMethods implements InterfaceWithDuplicateMethods {
    public int getZeroFromStaticInterfaceMethod() {
      return InterfaceWithDuplicateMethods.getZero(this);
    }
  }
}
