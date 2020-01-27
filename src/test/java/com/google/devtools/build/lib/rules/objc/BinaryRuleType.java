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

import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * This exists for convenience for any rule type requires the exact same attributes as
 * {@code objc_binary}. If some rule ever changes to require more or fewer attributes, it is OK
 * to stop using this class.
 */
final class BinaryRuleType extends RuleType {
  BinaryRuleType(String ruleTypeName) {
    super(ruleTypeName);
  }

  @Override
  Iterable<String> requiredAttributes(
      Scratch scratch, String packageName, Set<String> alreadyAdded) throws IOException {
    List<String> attributes = new ArrayList<>();
    if (!alreadyAdded.contains("srcs") && !alreadyAdded.contains("non_arc_srcs")) {
      scratch.file(packageName + "/a.m");
      scratch.file(packageName + "/private.h");
      attributes.add("srcs = ['a.m', 'private.h']");
    }
    return attributes;
  }
}
