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

import java.util.List;
import java.util.stream.Collectors;

public class MethodReferenceInSubclass extends MethodReferenceSuperclass {

  public MethodReferenceInSubclass(List<String> names) {
    super(names);
  }

  // Private method reference in subclass that causes a bridge method with the same signature as in
  // a superclass in the same package (regression test for b/36201257).  Both superclass and this
  // class need a method reference to a private *instance* method with the same signature, and they
  // should each only one method reference and no lambdas so any class-local counter matches, for
  // this class to serve as a repro for b/36201257.
  public List<String> containsE() {
    return names.stream().filter(this::containsE).collect(Collectors.toList());
  }

  private boolean containsE(String input) {
    return input.contains("e");
  }
}
