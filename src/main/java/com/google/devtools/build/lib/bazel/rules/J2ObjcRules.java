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
package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.bazel.rules.objc.BazelJ2ObjcLibraryRule;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.objc.J2ObjcLibraryBaseRule;

/**
 * Rules for supporting transpilation from Java to Objective-C in Bazel.
 */
public class J2ObjcRules implements RuleSet {
  public static final J2ObjcRules INSTANCE = new J2ObjcRules();

  private J2ObjcRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.addRuleDefinition(new J2ObjcLibraryBaseRule());
    builder.addRuleDefinition(new BazelJ2ObjcLibraryRule());
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE, JavaRules.INSTANCE,
              ObjcRules.INSTANCE);
  }
}
