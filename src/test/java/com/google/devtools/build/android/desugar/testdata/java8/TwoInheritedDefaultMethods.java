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

/** Desugar test class that explicitly calls default methods from two implemented interfaces. */
public class TwoInheritedDefaultMethods implements Name1, Name2 {
  @Override
  public String name() {
    return Name1.super.name() + ":" + Name2.super.name();
  }
}

/** Test interface for {@link TwoInheritedDefaultMethods}. */
interface Name1 {
  default String name() {
    return "One";
  }
}

/** Test interface for {@link TwoInheritedDefaultMethods}. */
interface Name2 {
  default String name() {
    return "Two";
  }
}
