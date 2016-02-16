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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for compiler definition.
 */
public final class CcToolchainRule implements RuleDefinition {
  private static final LateBoundLabel<BuildConfiguration> LIBC_LINK =
      new LateBoundLabel<BuildConfiguration>(CppConfiguration.class) {
        @Override
        public Label getDefault(Rule rule, AttributeMap attributes,
            BuildConfiguration configuration) {
          return configuration.getFragment(CppConfiguration.class).getLibcLabel();
        }
      };

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .setUndocumented()
        .requiresConfigurationFragments(CppConfiguration.class)
        .add(attr("output_licenses", LICENSE))
        .add(attr("cpu", STRING).mandatory())
        .add(attr("all_files", LABEL).legacyAllowAnyFileType().cfg(HOST).mandatory())
        .add(attr("compiler_files", LABEL).legacyAllowAnyFileType().cfg(HOST).mandatory())
        .add(attr("strip_files", LABEL).legacyAllowAnyFileType().cfg(HOST).mandatory())
        .add(attr("objcopy_files", LABEL).legacyAllowAnyFileType().cfg(HOST).mandatory())
        .add(attr("linker_files", LABEL).legacyAllowAnyFileType().cfg(HOST).mandatory())
        .add(attr("dwp_files", LABEL).legacyAllowAnyFileType().cfg(HOST).mandatory())
        .add(attr("static_runtime_libs", LABEL_LIST).legacyAllowAnyFileType().mandatory())
        .add(attr("dynamic_runtime_libs", LABEL_LIST).legacyAllowAnyFileType().mandatory())
        .add(attr("module_map", LABEL).legacyAllowAnyFileType().cfg(HOST))
        .add(attr("supports_param_files", BOOLEAN).value(true))
        .add(attr("supports_header_parsing", BOOLEAN).value(false))
        // TODO(bazel-team): Should be using the TARGET configuration.
        .add(attr(":libc_link", LABEL).cfg(HOST).value(LIBC_LINK))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_toolchain")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(CcToolchain.class)
        .build();
  }
}
