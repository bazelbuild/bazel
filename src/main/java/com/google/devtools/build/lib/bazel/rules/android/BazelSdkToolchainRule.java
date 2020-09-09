// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.rules.android.AndroidSdkProvider.ANDROID_SDK_TOOLCHAIN_TYPE_ATTRIBUTE_NAME;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

/** Rule for accessing the android sdk via toolchains. */
public final class BazelSdkToolchainRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
    // This lives in tools/android/BUILD.tools
    Label toolchainType = env.getToolsLabel("//tools/android:sdk_toolchain_type");

    return builder
        .addRequiredToolchains(toolchainType)
        .add(attr(ANDROID_SDK_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL).value(toolchainType))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$bazel_sdk_toolchain")
        .type(RuleClassType.ABSTRACT)
        .build();
  }
}
