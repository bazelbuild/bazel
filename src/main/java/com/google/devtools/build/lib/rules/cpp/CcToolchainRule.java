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
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Rule definition for compiler definition.
 */
public final class CcToolchainRule implements RuleDefinition {

  public static final String LIBC_TOP_ATTR = ":libc_top";
  public static final String FDO_OPTIMIZE_ATTR = ":fdo_optimize";
  public static final String FDO_PROFILE_ATTR = ":fdo_profile";
  public static final String XFDO_PROFILE_ATTR = ":xfdo_profile";
  public static final String TOOLCHAIN_CONFIG_ATTR = "toolchain_config";

  /**
   * Determines if the given target is a cc_toolchain or one of its subclasses. New subclasses
   * should be added to this method.
   */
  static boolean isCcToolchain(Target target) {
    String ruleClass = ((Rule) target).getRuleClass();
    return ruleClass.endsWith("cc_toolchain");
  }

  private static Label getLabel(AttributeMap attributes, String attrName, Label defaultValue) {
    if (attributes.has(attrName, LABEL)) {
      Label value = attributes.get(attrName, LABEL);
      if (value != null) {
        return value;
      }
    }
    return defaultValue;
  }

  private static final LabelLateBoundDefault<?> LIBC_TOP_VALUE =
      LabelLateBoundDefault.fromTargetConfiguration(
          CppConfiguration.class,
          null,
          (rule, attributes, cppConfig) -> {
            // Is the libcTop directly set via a flag?
            Label cppOptionLibcTop = cppConfig.getLibcTopLabel();
            if (cppOptionLibcTop != null) {
              return cppOptionLibcTop;
            }

            // Look up the value from the attribute.
            // This avoids analyzing the label from the CROSSTOOL if the attribute is set.
            return getLabel(attributes, "libc_top", /* defaultValue= */ null);
          });

  private static final LabelLateBoundDefault<?> FDO_OPTIMIZE_VALUE =
      LabelLateBoundDefault.fromTargetConfiguration(
          CppConfiguration.class,
          null,
          (rule, attributes, cppConfig) -> cppConfig.getFdoOptimizeLabel());

  private static final LabelLateBoundDefault<?> FDO_PROFILE_VALUE =
      LabelLateBoundDefault.fromTargetConfiguration(
          CppConfiguration.class,
          null,
          (rule, attributes, cppConfig) -> cppConfig.getFdoProfileLabel());

  private static final LabelLateBoundDefault<?> XFDO_PROFILE_VALUE =
      LabelLateBoundDefault.fromTargetConfiguration(
          CppConfiguration.class,
          null,
          (rule, attributes, cppConfig) -> cppConfig.getXFdoProfileLabel());

  private static final LabelLateBoundDefault<?> FDO_PREFETCH_HINTS =
      LabelLateBoundDefault.fromTargetConfiguration(
      CppConfiguration.class,
      null,
      (rule, attributes, cppConfig) -> cppConfig.getFdoPrefetchHintsLabel());

  /**
   * Returns true if zipper should be loaded. We load the zipper executable if FDO optimization is
   * enabled through --fdo_optimize or --fdo_profile
   */
  private static boolean shouldIncludeZipperInToolchain(CppConfiguration cppConfiguration) {
    return cppConfiguration.getFdoOptimizeLabel() != null
        || cppConfiguration.getFdoProfileLabel() != null
        || cppConfiguration.getFdoPath() != null;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    final Label zipper = env.getToolsLabel("//tools/zip:zipper");
    return builder
        .setUndocumented()
        .requiresConfigurationFragments(CppConfiguration.class, PlatformConfiguration.class)
        .advertiseProvider(TemplateVariableInfo.class)
        .add(attr("output_licenses", LICENSE))
        .add(attr("cpu", STRING).nonconfigurable("Used in configuration creation").mandatory())
        .add(attr("compiler", STRING).nonconfigurable("Used in configuration creation"))
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
            attr("compiler_files_without_includes", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE))
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
                        (rule, attributes, cppConfig) ->
                            shouldIncludeZipperInToolchain(cppConfig) ? zipper : null)))

        // TODO(b/78578234): Make this the default and remove the late-bound versions.
        .add(attr("libc_top", LABEL).allowedFileTypes())
        .add(attr(LIBC_TOP_ATTR, LABEL).value(LIBC_TOP_VALUE))
        .add(attr(FDO_OPTIMIZE_ATTR, LABEL).value(FDO_OPTIMIZE_VALUE))
        .add(
            attr(XFDO_PROFILE_ATTR, LABEL)
                .allowedRuleClasses("fdo_profile")
                .mandatoryProviders(ImmutableList.of(FdoProfileProvider.PROVIDER.id()))
                .value(XFDO_PROFILE_VALUE))
        .add(
            attr(FDO_PROFILE_ATTR, LABEL)
                .allowedRuleClasses("fdo_profile")
                .mandatoryProviders(ImmutableList.of(FdoProfileProvider.PROVIDER.id()))
                .value(FDO_PROFILE_VALUE))
        .add(
            attr(":fdo_prefetch_hints", LABEL)
                .allowedRuleClasses("fdo_prefetch_hints")
                .mandatoryProviders(ImmutableList.of(FdoPrefetchHintsProvider.PROVIDER.id()))
                .value(FDO_PREFETCH_HINTS))
        .add(attr("proto", Type.STRING))
        .add(
            attr("toolchain_identifier", Type.STRING)
                .nonconfigurable("Used in configuration creation")
                .value(""))
        .add(
            attr(TOOLCHAIN_CONFIG_ATTR, LABEL)
                .allowedFileTypes()
                .mandatoryProviders(CcToolchainConfigInfo.PROVIDER.id()))
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
