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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.Set;

/**
 * This exists for convenience for any rule type that requires only one file in {@code srcs} or
 * {@code non_arc_srcs}, and no other attributes.
 */
final class OnlyNeedsSourcesRuleType extends RuleType {
  OnlyNeedsSourcesRuleType(String ruleTypeName) {
    super(ruleTypeName);
  }

  @Override
  Iterable<String> requiredAttributes(
      Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
    ImmutableList.Builder<String> attributes = new ImmutableList.Builder<>();
    if (!alreadyAdded.contains("srcs") && !alreadyAdded.contains("non_arc_srcs")) {
      scratch.file(packageDir + "/a.m");
      scratch.file(packageDir + "/private.h");
      attributes.add("srcs = ['a.m', 'private.h']");
    }
    return attributes.build();
  }
}
