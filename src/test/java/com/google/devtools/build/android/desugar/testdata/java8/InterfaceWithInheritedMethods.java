// Copyright 2018 The Bazel Authors. All rights reserved.
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

/** Regression test data for b/73355452 that also includes calling static methods. */
public interface InterfaceWithInheritedMethods {
  default String name() {
    return "Base";
  }

  static String staticSuffix() {
    return "!";
  }

  static interface Passthrough extends InterfaceWithInheritedMethods {
    // inherits name().  Note that desugar doesn't produce a companion class for this interface
    // since it doesn't define any default or static interface methods itself.
  }

  static class Impl implements Passthrough {
    @Override
    public String name() {
      // Even though Passthrough itself doesn't define name(), bytecode refers to Passthrough.name.
      return Passthrough.super.name();
    }

    public String suffix() {
      // Note that Passthrough.defaultSuffix doesn't compile and bytecode refers to
      // InterfaceWithInheritedMethods.staticSuffix, so this shouldn't cause issues like b/73355452
      return staticSuffix();
    }
  }
}
