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
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.workspace.WorkspaceBaseRule;
import com.google.devtools.build.lib.bazel.rules.workspace.WorkspaceConfiguredTargetFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * Definitions for the Android repository WORKSPACE rules.
 */
public class AndroidRepositoryRules {
    private static final ImmutableList<String> TOOLS = ImmutableList.of(
        "proguard_whitelister",
        "merge_manifests",
        "build_incremental_dexmanifest",
        "stubify_manifest",
        "incremental_install",
        "build_split_manifest",
        "strip_resources",
        "incremental_stub_application",
        "incremental_split_stub_application",
        "resources_processor",
        "aar_generator",
        "shuffle_jars",
        "merge_dexzips",
        "debug_keystore",
        "IdlClass");

    private static final Function<? super Rule, Map<String, Label>> BINDINGS_FUNCTION =
        new Function< Rule, Map<String, Label>>() {
          @Nullable
          @Override
          public Map<String, Label> apply(Rule rule) {
            ImmutableMap.Builder<String, Label> result = ImmutableMap.builder();
            for (String tool : TOOLS) {
              result.put("android/" + tool, Label.parseAbsoluteUnchecked(
                  "@" + rule.getName() + "//tools/android:" + tool));
            }

            return result.build();
          }
        };

  /**
   * Definition of the {@code android_local_tools_repository} rule.
   */
  public static class AndroidLocalRepositoryRule implements RuleDefinition {
    public static final String NAME = "android_local_tools_repository";

    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .setUndocumented()
          .setWorkspaceOnly()
          .setExternalBindingsFunction(BINDINGS_FUNCTION)
          .add(attr("path", STRING).mandatory().nonconfigurable("WORKSPACE rule"))
          .build();

    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name(AndroidLocalRepositoryRule.NAME)
          .type(RuleClassType.WORKSPACE)
          .ancestors(WorkspaceBaseRule.class)
          .factoryClass(WorkspaceConfiguredTargetFactory.class)
          .build();

    }
  }

  /**
   * Definition of the {@code android_http_tools_repository} rule.
   */
  public static class AndroidHttpToolsRepositoryRule implements RuleDefinition {
    public static final String NAME = "android_http_tools_repository";

    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .setUndocumented()
          .setWorkspaceOnly()
          .setExternalBindingsFunction(BINDINGS_FUNCTION)
          .add(attr("url", STRING).mandatory())
          .add(attr("sha256", STRING).mandatory())
          .build();

    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name(AndroidHttpToolsRepositoryRule.NAME)
          .type(RuleClassType.WORKSPACE)
          .ancestors(WorkspaceBaseRule.class)
          .factoryClass(WorkspaceConfiguredTargetFactory.class)
          .build();

    }
  }

  /**
   * List of tools for {@link com.google.devtools.build.lib.analysis.mock.BazelAnalysisMock}.
   */
  @VisibleForTesting
  public static ImmutableList<String> toolsForTesting() {
    return TOOLS;
  }

}
