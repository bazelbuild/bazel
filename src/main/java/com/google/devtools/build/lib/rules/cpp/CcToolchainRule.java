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
import com.google.devtools.build.lib.packages.RuleClass;
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
        .requiresConfigurationFragments(CppConfiguration.class, PlatformConfiguration.class)
        .advertiseProvider(TemplateVariableInfo.class)
        .add(attr("output_licenses", LICENSE))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(cpu) -->
         Deprecated. Use toolchain_identifier attribute instead. It will be a noop after
         <a href="https://github.com/bazelbuild/bazel/issues/5380">CROSSTOOL migration to Starlark
         </a>.

         <p>When set, it will be used to perform crosstool_config.toolchain selection. It will take
         precedence over --cpu Bazel option.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("cpu", STRING).nonconfigurable("Used in configuration creation").mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(compiler) -->
         Deprecated. Use toolchain_identifier attribute instead. It will be a noop after
         <a href="https://github.com/bazelbuild/bazel/issues/5380">CROSSTOOL migration to Starlark
         </a>.

         <p>When set, it will be used to perform crosstool_config.toolchain selection. It will take
         precedence over --cpu Bazel option.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("compiler", STRING).nonconfigurable("Used in configuration creation"))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(all_files) -->
        Collection of all cc_toolchain artifacts. These artifacts will be added as inputs to all
        rules_cc related actions (with the exception of actions that are using more precise sets of
        artifacts from attributes below).

        <p>This is what Runfiles provider of the cc_toolchain contains.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("all_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(compiler_files) -->
        Collection of all cc_toolchain artifacts required for compile actions.

        <p>Currently only used by lto_backend actions, regular compile actions use all_files
         (<a href="https://github.com/bazelbuild/bazel/issues/6927">#6927</a>).
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("compiler_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(compiler_files_without_includes) -->
        Collection of all cc_toolchain artifacts required for compile actions in case when
        input discovery is supported (currently Google-only).
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("compiler_files_without_includes", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(strip_files) -->
        Collection of all cc_toolchain artifacts required for strip actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("strip_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(objcopy_files) -->
        Collection of all cc_toolchain artifacts required for objcopy actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("objcopy_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(as_files) -->
        Currently unused (<a href="https://github.com/bazelbuild/bazel/issues/6928">#6928</a>).

        <p>Collection of all cc_toolchain artifacts required for assembly actions.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("as_files", LABEL).legacyAllowAnyFileType().cfg(HostTransition.INSTANCE))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(as_files) -->
        Currently unused (<a href="https://github.com/bazelbuild/bazel/issues/6928">#6928</a>).

        <p>Collection of all cc_toolchain artifacts required for archiving actions.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("ar_files", LABEL).legacyAllowAnyFileType().cfg(HostTransition.INSTANCE))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(linker_files) -->
        Collection of all cc_toolchain artifacts required for linking actions.

        <p>Currently also used for archiving actions
         (<a href="https://github.com/bazelbuild/bazel/issues/6928">#6928</a>).</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("linker_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(dwp_files) -->
        Collection of all cc_toolchain artifacts required for dwp actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("dwp_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(HostTransition.INSTANCE)
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(coverage_files) -->
        Collection of all cc_toolchain artifacts required for coverage actions. If not specified,
        all_files are used.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("coverage_files", LABEL).legacyAllowAnyFileType().cfg(HostTransition.INSTANCE))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(static_runtime_libs) -->
        A collection of artifacts for static libraries for the C++ runtime library
        (e.g. libstdc++.a).

        <p>cc_toolchain will select one of these libraries based on the label from
        crosstool_proto.static_runtimes_filegroup field.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("static_runtime_libs", LABEL_LIST).legacyAllowAnyFileType())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(dynamic_runtime_libs) -->
        A collection of artifacts for dynamic libraries for the C++ runtime library
        (e.g. libstdc++.so).

        <p>cc_toolchain will select one of these libraries based on the label from
        crosstool_proto.dynamic_runtimes_filegroup field.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("dynamic_runtime_libs", LABEL_LIST).legacyAllowAnyFileType())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(module_map) -->
        Module map artifact to be used for modular builds.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("module_map", LABEL).legacyAllowAnyFileType().cfg(HostTransition.INSTANCE))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(supports_param_files) -->
        Set to True when cc_toolchain supports using param files for linking actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("supports_param_files", BOOLEAN).value(true))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(supports_header_parsing) -->
        Set to True when cc_toolchain supports header parsing actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
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
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(libc_top) -->
        A collection of artifacts for libc passed as inputs to compile/linking actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
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
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(proto) -->
        Inline content of the crosstool_config.toolchain. When specified, it will take precedence
        over the CROSSTOOL file.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("proto", Type.STRING))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(toolchain_identifier) -->
        Deprecated. Will be replaced by toolchain_config attribute.

        <p>The identifier used to match this cc_toolchain with the corresponding
        crosstool_config.toolchain.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("toolchain_identifier", Type.STRING)
                .nonconfigurable("Used in configuration creation")
                .value(""))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(toolchain_config) -->
        The label of the rule providing cc_toolchain_config_info. When specified, cc_toolchain will
        not use the CROSSTOOL file at all.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
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
