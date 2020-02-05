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

import com.google.common.collect.ImmutableList;
import java.util.List;
import java.util.stream.Collectors;

public class ConcreteDefaultInterfaceWithLambda implements DefaultInterfaceWithLambda {
  static final String ONE = String.valueOf(1);

  @Override
  public List<String> digits() {
    return ImmutableList.of(0, 2).stream()
        .map(i -> i == 0 ? ONE : String.valueOf(i))
        .collect(Collectors.toList());
  }
}
