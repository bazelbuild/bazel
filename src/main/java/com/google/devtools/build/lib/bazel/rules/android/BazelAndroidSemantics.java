// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidSemantics;

/**
 * Implementation of Bazel-specific behavior in Android rules.
 */
public class BazelAndroidSemantics implements AndroidSemantics {
  public static final BazelAndroidSemantics INSTANCE = new BazelAndroidSemantics();

  private static final ImmutableSet<PackageIdentifier> STARLARK_MIGRATION_NATIVE_USAGE_ALLOW_LIST =
      // Internal package identifiers that are allowed to use the native Android rules until they
      // can be fully moved into the rules_android Starlark implementation.
      ImmutableSet.<PackageIdentifier>builder()
          .add(PackageIdentifier.createUnchecked("bazel_tools", "tools/android"))
          .build();

  private BazelAndroidSemantics() {}

  @Override
  public String getNativeDepsFileName() {
    return "nativedeps";
  }

  @Override
  public ImmutableList<String> getCompatibleJavacOptions(RuleContext ruleContext) {
    ImmutableList.Builder<String> javacArgs = new ImmutableList.Builder<>();
    if (!ruleContext.getFragment(AndroidConfiguration.class).desugarJava8()) {
      javacArgs.add("-source", "7", "-target", "7");
    }
    return javacArgs.build();
  }

  @Override
  public void registerMigrationRuleError(RuleContext ruleContext) throws RuleErrorException {
    if (STARLARK_MIGRATION_NATIVE_USAGE_ALLOW_LIST.contains(
        ruleContext.getLabel().getPackageIdentifier())) {
      return;
    }

    ruleContext.attributeError(
        "tags",
        "The native Android rules are deprecated. Please use the Starlark Android rules by adding "
            + "the following load statement to the BUILD file: "
            + "load(\"@build_bazel_rules_android//android:rules.bzl\", \""
            + ruleContext.getRule().getRuleClass()
            + "\"). See http://github.com/bazelbuild/rules_android.");
  }
}
