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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import javax.annotation.Nullable;

/**
 * Implementation for the {@code android_tools_defaults_jar} rule.
 *
 * <p>This rule is a sad, sad way to let people depend on {@code android.jar} when an {@code
 * android_sdk} rule is used. In an ideal world, people would say "depend on the android_jar output
 * group of $config.android_sdk", but, alas, neither depending on labels in the configuration nor
 * depending on a specified output group works. So all this needs to be implemented manually.
 */
public class AndroidToolsDefaultsJar implements RuleConfiguredTargetFactory {

  private final AndroidSemantics androidSemantics;

  protected AndroidToolsDefaultsJar(AndroidSemantics androidSemantics) {
    this.androidSemantics = androidSemantics;
  }

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    androidSemantics.checkForMigrationTag(ruleContext);
    Artifact androidJar = AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar();

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .setFilesToBuild(NestedSetBuilder.create(Order.STABLE_ORDER, androidJar))
        .build();
  }
}
