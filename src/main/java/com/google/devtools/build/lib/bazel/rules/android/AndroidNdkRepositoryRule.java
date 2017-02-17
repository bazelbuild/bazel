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
import static com.google.devtools.build.lib.syntax.Type.INTEGER;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpls;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceConfiguredTargetFactory;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Definition of the {@code android_ndk_repository} rule.
 */
public class AndroidNdkRepositoryRule implements RuleDefinition {
  public static final String NAME = "android_ndk_repository";

  private static final Function<? super Rule, Map<String, Label>> BINDINGS_FUNCTION =
      new Function< Rule, Map<String, Label>>() {
        @Nullable
        @Override
        public Map<String, Label> apply(Rule rule) {

          String defaultToolchainName =
              AndroidNdkRepositoryFunction.createToolchainName(StlImpls.DEFAULT_STL_NAME);

          return ImmutableMap.of(
              "android/crosstool",
              Label.parseAbsoluteUnchecked("@" + rule.getName() + "//:" + defaultToolchainName),
              "android_ndk_for_testing",
              Label.parseAbsoluteUnchecked("@" + rule.getName() + "//:files"));
        }
      };

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setUndocumented()
        .setWorkspaceOnly()
        .setExternalBindingsFunction(BINDINGS_FUNCTION)
        .add(attr("path", STRING).nonconfigurable("WORKSPACE rule"))
        .add(attr("api_level", INTEGER).nonconfigurable("WORKSPACE rule"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(AndroidNdkRepositoryRule.NAME)
        .type(RuleClassType.WORKSPACE)
        .ancestors(WorkspaceBaseRule.class)
        .factoryClass(WorkspaceConfiguredTargetFactory.class)
        .build();
  }
}
