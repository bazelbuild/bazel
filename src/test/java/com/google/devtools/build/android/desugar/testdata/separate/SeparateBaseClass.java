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
package com.google.devtools.build.android.desugar.testdata.separate;

import java.util.List;

/** Test base class for testing method references to protected methods in another compilation. */
public class SeparateBaseClass<T> {

  private final List<T> list;

  protected SeparateBaseClass(List<T> list) {
    this.list = list;
  }

  protected boolean contains(T elem) {
    return list.contains(elem);
  }
}
