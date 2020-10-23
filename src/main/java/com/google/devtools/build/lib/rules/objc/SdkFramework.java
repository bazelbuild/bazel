// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;

/**
 * Represents the name of an SDK framework.
 * <p>
 * Besides being a glorified String, this class prevents you from adding framework names to an
 * argument list without explicitly specifying how to prefix them.
 */
final class SdkFramework extends Value<SdkFramework> {
  private final String name;

  public SdkFramework(String name) {
    super(name);
    this.name = name;
  }

  public String getName() {
    return name;
  }

  /** Returns an iterable which contains the name of each given framework in the same order. */
  static ImmutableList<String> names(NestedSet<SdkFramework> frameworks) {
    ImmutableList.Builder<String> result = new ImmutableList.Builder<>();
    for (SdkFramework framework : frameworks.toList()) {
      result.add(framework.getName());
    }
    return result.build();
  }
}
