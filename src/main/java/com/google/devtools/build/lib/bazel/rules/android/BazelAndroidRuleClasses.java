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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;
import com.google.devtools.build.lib.rules.android.AndroidToolsDefaultsJar;

/** Rule class definitions for Android rules. */
public class BazelAndroidRuleClasses {

  /** Definition of the {@code android_tools_defaults_jar} rule. */
  public static final class BazelAndroidToolsDefaultsJarRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .setUndocumented()
          .add(
              attr(":android_sdk", LABEL)
                  .allowedRuleClasses("android_sdk", "filegroup")
                  .value(AndroidRuleClasses.ANDROID_SDK))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("android_tools_defaults_jar")
          .ancestors(BaseRuleClasses.BaseRule.class)
          .factoryClass(AndroidToolsDefaultsJar.class)
          .build();
    }
  }
}
