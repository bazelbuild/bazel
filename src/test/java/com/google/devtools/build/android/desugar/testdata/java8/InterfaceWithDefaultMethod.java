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

/** Interface for testing default methods overridden by extending interfaces. */
public interface InterfaceWithDefaultMethod {
  default int version() {
    return 1;
  }

  /** Interface that overrides {@link #version}. */
  public interface Redefine extends InterfaceWithDefaultMethod {
    @Override
    default int version() {
      return 2;
    }
  }

  /** Class that implements both interfaces, supertype before subtype. */
  public static class Version2 implements InterfaceWithDefaultMethod, Redefine {}

  /** Base class that just implements {@link Redefine}. */
  static class Version2Base implements Redefine {}

  /**
   * Subclass that implements an interface explicitly that the superclass also implements, but the
   * superclass implements a more specific interface that overrides a defautl method.
   */
  public static class AlsoVersion2 extends Version2Base implements InterfaceWithDefaultMethod {}
}
