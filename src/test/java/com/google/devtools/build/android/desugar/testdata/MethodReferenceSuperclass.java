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

public class MethodReferenceSuperclass {

  protected final List<String> names;

  public MethodReferenceSuperclass(List<String> names) {
    this.names = names;
  }

  // Method reference that causes a simple bridge method because the referenced method is private.
  // We want to make sure that bridge methods generated in subclasses don't clobber this one.
  public List<String> startsWithL() {
    return names.stream().filter(this::startsWithL).collect(Collectors.toList());
  }

  private boolean startsWithL(String input) {
    return input.startsWith("L");
  }
}
