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
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.cpp.transitions.LipoContextCollectorTransition;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Rule definition for compiler definition.
 */
public final class CcToolchainRule implements RuleDefinition {
  /**
   * Determines if the given target is a cc_toolchain or one of its subclasses. New subclasses
   * should be added to this method.
   */
  static boolean isCcToolchain(Target target) {
    String ruleClass = ((Rule) target).getRuleClass();
    return ruleClass.endsWith("cc_toolchain");
  }

  private static final LabelLateBoundDefault<?> LIBC_TOP =
      LabelLateBoundDefault.fromTargetConfiguration(
          CppConfiguration.class,
          null,
          (rule, attributes, cppConfig) -> cppConfig.getSysrootLabel());

  private static final LabelLateBoundDefault<?> FDO_OPTIMIZE_LABEL =
      LabelLateBoundDefault.fromTargetConfiguration(
          CppConfiguration.class,
          null,
          (rule, attributes, cppConfig) -> cppConfig.getFdoOptimizeLabel());

  private static final LabelLateBoundDefault<?> FDO_PROFILE_LABEL =
      LabelLateBoundDefault.fromTargetConfiguration(
          CppConfiguration.class,
          null,
          (rule, attributes, cppConfig) -> cppConfig.getFdoProfileLabel());

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    final Label zipper = env.getToolsLabel("//tools/zip:zipper");
    return builder
        .setUndocumented()
        .requiresConfigurationFragments(CppConfiguration.class, PlatformConfiguration.class)
        .advertiseProvider(TemplateVariableInfo.class)
        .add(attr("output_licenses", LICENSE))
        .add(attr("cpu", STRING).mandatory())
        .add(attr("compiler", STRING))
        .add(attr("libc", STRING))
        .add(
            attr("all_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        .add(
            attr("compiler_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        .add(
            attr("strip_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        .add(
            attr("objcopy_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        .add(attr("as_files", LABEL).legacyAllowAnyFileType().cfg(HostTransition.INSTANCE))
        .add(attr("ar_files", LABEL).legacyAllowAnyFileType().cfg(HostTransition.INSTANCE))
        .add(
            attr("linker_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        .add(
            attr("dwp_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        .add(attr("coverage_files", LABEL).legacyAllowAnyFileType().cfg(HostTransition.INSTANCE))
        .add(attr("static_runtime_libs", LABEL_LIST).legacyAllowAnyFileType().mandatory())
        .add(attr("dynamic_runtime_libs", LABEL_LIST).legacyAllowAnyFileType().mandatory())
        .add(attr("module_map", LABEL).legacyAllowAnyFileType().cfg(HostTransition.INSTANCE))
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
            attr(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL)
                .value(CppRuleClasses.ccToolchainTypeAttribute(env)))
        .add(
            attr(":zipper", LABEL)
                .cfg(HostTransition.INSTANCE)
                .singleArtifact()
                .value(
                    LabelLateBoundDefault.fromTargetConfiguration(
                        CppConfiguration.class,
                        null,
                        // TODO(b/69547565): Remove call to shouldIncludeZipperInToolchain
                        (rule, attributes, cppConfig) ->
                            cppConfig.shouldIncludeZipperInToolchain() ? zipper : null)))
        .add(attr(":libc_top", LABEL).value(LIBC_TOP))
        .add(attr(":fdo_optimize", LABEL).singleArtifact().value(FDO_OPTIMIZE_LABEL))
        .add(
            attr(":fdo_profile", LABEL)
                .allowedRuleClasses("fdo_profile")
                .mandatoryProviders(ImmutableList.of(FdoProfileProvider.PROVIDER.id()))
                .value(FDO_PROFILE_LABEL))
        .add(
            attr(TransitiveLipoInfoProvider.LIPO_CONTEXT_COLLECTOR, LABEL)
                .cfg(LipoContextCollectorTransition.INSTANCE)
                .value(CppRuleClasses.LIPO_CONTEXT_COLLECTOR)
                .skipPrereqValidatorCheck())
        .add(attr("proto", Type.STRING))
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
