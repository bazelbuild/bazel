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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.cpp.transitions.LipoContextCollectorTransition;

/**
 * Rule definition for compiler definition.
 */
public final class CcToolchainRule implements RuleDefinition {

  /**
   * The label points to the Windows object file parser. In bazel, it should be
   * //tools/def_parser:def_parser, otherwise it should be null.
   *
   * <p>TODO(pcloudy): Remove this after Bazel rule definitions are not used internally anymore.
   * Related bug b/63658220
   */
  private final String defParserLabel;

  public CcToolchainRule(String defParser) {
    this.defParserLabel = defParser;
  }

  public CcToolchainRule() {
    this.defParserLabel = null;
  }

  /**
   * Determines if the given target is a cc_toolchain or one of its subclasses. New subclasses
   * should be added to this method.
   */
  static boolean isCcToolchain(Target target) {
    String ruleClass = ((Rule) target).getRuleClass();
    return ruleClass.endsWith("cc_toolchain");
  }

  private static final LateBoundDefault<?, Label> LIBC_TOP =
      LateBoundDefault.fromTargetConfiguration(
          CppConfiguration.class,
          null,
          (rule, attributes, cppConfig) -> cppConfig.getSysrootLabel());

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    final Label zipper = env.getToolsLabel("//tools/zip:zipper");
    if (defParserLabel != null) {
      builder.add(
          attr("$def_parser", LABEL)
              .cfg(HostTransition.INSTANCE)
              .singleArtifact()
              .value(env.getLabel(defParserLabel)));
    }
    return builder
        .setUndocumented()
        .requiresConfigurationFragments(CppConfiguration.class, PlatformConfiguration.class)
        .advertiseProvider(TemplateVariableInfo.class)
        .add(attr("output_licenses", LICENSE))
        .add(attr("cpu", STRING).mandatory())
        .add(attr("compiler", STRING))
        .add(attr("libc", STRING))
        .add(attr("all_files", LABEL)
            .legacyAllowAnyFileType()
            .cfg(HostTransition.INSTANCE)
            .mandatory())
        .add(attr("compiler_files", LABEL)
            .legacyAllowAnyFileType()
            .cfg(HostTransition.INSTANCE)
            .mandatory())
        .add(attr("strip_files", LABEL)
            .legacyAllowAnyFileType()
            .cfg(HostTransition.INSTANCE)
            .mandatory())
        .add(attr("objcopy_files", LABEL)
            .legacyAllowAnyFileType()
            .cfg(HostTransition.INSTANCE)
            .mandatory())
        .add(attr("linker_files", LABEL)
            .legacyAllowAnyFileType()
            .cfg(HostTransition.INSTANCE)
            .mandatory())
        .add(attr("dwp_files", LABEL)
            .legacyAllowAnyFileType()
            .cfg(HostTransition.INSTANCE)
            .mandatory())
        .add(attr("coverage_files", LABEL)
            .legacyAllowAnyFileType()
            .cfg(HostTransition.INSTANCE))
        .add(attr("static_runtime_libs", LABEL_LIST)
            .legacyAllowAnyFileType()
            .mandatory())
        .add(attr("dynamic_runtime_libs", LABEL_LIST)
            .legacyAllowAnyFileType()
            .mandatory())
        .add(attr("module_map", LABEL)
            .legacyAllowAnyFileType()
            .cfg(HostTransition.INSTANCE))
        .add(attr("supports_param_files", BOOLEAN).value(true))
        .add(attr("supports_header_parsing", BOOLEAN).value(false))
        .add(
            attr("$interface_library_builder", LABEL)
                .cfg(HostTransition.INSTANCE)
                .singleArtifact()
                .value(env.getToolsLabel("//tools/cpp:interface_library_builder")))
        .add(
            attr("$link_dynamic_library_tool", LABEL)
                .cfg(HostTransition.INSTANCE)
                .singleArtifact()
                .value(env.getToolsLabel("//tools/cpp:link_dynamic_library")))
        .add(
            attr(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, LABEL)
                .value(CppRuleClasses.ccToolchainTypeAttribute(env)))
        .add(
            attr(":zipper", LABEL)
                .cfg(HostTransition.INSTANCE)
                .singleArtifact()
                .value(
                    LateBoundDefault.fromTargetConfiguration(
                        CppConfiguration.class,
                        null,
                        // TODO(b/69547565): Remove call to isLLVMOptimizedFdo
                        (rule, attributes, cppConfig) ->
                            cppConfig.isLLVMOptimizedFdo() ? zipper : null)))
        .add(attr(":libc_top", LABEL).value(LIBC_TOP))
        .add(
            attr(":lipo_context_collector", LABEL)
                .cfg(LipoContextCollectorTransition.INSTANCE)
                .value(CppRuleClasses.LIPO_CONTEXT_COLLECTOR)
                .skipPrereqValidatorCheck())
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
