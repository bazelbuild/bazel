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
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Type;

/**
 * Rule definition for compiler definition.
 */
public final class CcToolchainRule implements RuleDefinition {

  public static final String TOOLCHAIN_CONFIG_ATTR = "toolchain_config";

  /** Default attribute name where rules store the reference to cc_toolchain */
  public static final String CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME = ":cc_toolchain";

  public static final String CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME_FOR_STARLARK = "$cc_toolchain";

  /** Default attribute name for the c++ toolchain type */
  public static final String CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME = "$cc_toolchain_type";

  public static final String ALLOWED_LAYERING_CHECK_FEATURES_ALLOWLIST =
      "disabling_parse_headers_and_layering_check_allowed";
  public static final String ALLOWED_LAYERING_CHECK_FEATURES_TARGET =
      "//tools/build_defs/cc/whitelists/parse_headers_and_layering_check:"
          + ALLOWED_LAYERING_CHECK_FEATURES_ALLOWLIST;
  public static final String LOOSE_HEADER_CHECK_ALLOWLIST =
      "loose_header_check_allowed_in_toolchain";

  public static final String FDO_OPTIMIZE_ATTR = ":fdo_optimize";
  public static final String FDO_PROFILE_ATTR = ":fdo_profile";
  public static final String CSFDO_PROFILE_ATTR = ":csfdo_profile";
  public static final String XFDO_PROFILE_ATTR = ":xfdo_profile";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class, PlatformConfiguration.class)
        .advertiseProvider(TemplateVariableInfo.class)
        .add(attr("output_licenses", LICENSE))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(cpu) -->
         Deprecated. Use toolchain_identifier attribute instead. It will be a noop after
         <a href="https://github.com/bazelbuild/bazel/issues/5380">CROSSTOOL migration to Starlark
         </a>, and will be removed by
         <a href="https://github.com/bazelbuild/bazel/issues/7075">#7075</a>.

         <p>When set, it will be used to perform crosstool_config.toolchain selection. It will take
         precedence over --cpu Bazel option.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("cpu", STRING).nonconfigurable("Used in configuration creation"))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(compiler) -->
         Deprecated. Use <code>toolchain_identifier</code> attribute instead. It will be a noop
         after
         <a href="https://github.com/bazelbuild/bazel/issues/5380">
           CROSSTOOL migration to Starlark
         </a>, and will be removed by
         <a href="https://github.com/bazelbuild/bazel/issues/7075">#7075</a>.

         <p>When set, it will be used to perform crosstool_config.toolchain selection. It will take
         precedence over --cpu Bazel option.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("compiler", STRING).nonconfigurable("Used in configuration creation"))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(all_files) -->
        Collection of all cc_toolchain artifacts. These artifacts will be added as inputs to all
        rules_cc related actions (with the exception of actions that are using more precise sets of
        artifacts from attributes below). Bazel assumes that <code>all_files</code> is a superset
        of all other artifact-providing attributes (e.g. linkstamp compilation needs both compile
        and link files, so it takes <code>all_files</code>).

        <p>
        This is what <code>cc_toolchain.files</code> contains, and this is used by all Starlark
        rules using C++ toolchain.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("all_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory())
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(compiler_files) -->
        Collection of all cc_toolchain artifacts required for compile actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("compiler_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory())
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(compiler_files_without_includes) -->
        Collection of all cc_toolchain artifacts required for compile actions in case when
        input discovery is supported (currently Google-only).
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("compiler_files_without_includes", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory()))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(strip_files) -->
        Collection of all cc_toolchain artifacts required for strip actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("strip_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory())
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(objcopy_files) -->
        Collection of all cc_toolchain artifacts required for objcopy actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("objcopy_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory())
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(as_files) -->
        <p>Collection of all cc_toolchain artifacts required for assembly actions.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("as_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory()))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(ar_files) -->
        <p>Collection of all cc_toolchain artifacts required for archiving actions.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("ar_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory()))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(linker_files) -->
        Collection of all cc_toolchain artifacts required for linking actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("linker_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory())
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(dwp_files) -->
        Collection of all cc_toolchain artifacts required for dwp actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("dwp_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory())
                .mandatory())
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(coverage_files) -->
        Collection of all cc_toolchain artifacts required for coverage actions. If not specified,
        all_files are used.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("coverage_files", LABEL)
                .legacyAllowAnyFileType()
                .cfg(new CcToolchainInputsTransitionFactory()))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(static_runtime_lib) -->
        Static library artifact for the C++ runtime library (e.g. libstdc++.a).

        <p>This will be used when 'static_link_cpp_runtimes' feature is enabled, and we're linking
        dependencies statically.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("static_runtime_lib", LABEL)
                .legacyAllowAnyFileType()
                // Should be in the target configuration
                .cfg(NoTransition.createFactory()))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(dynamic_runtime_lib) -->
        Dynamic library artifact for the C++ runtime library (e.g. libstdc++.so).

        <p>This will be used when 'static_link_cpp_runtimes' feature is enabled, and we're linking
        dependencies dynamically.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("dynamic_runtime_lib", LABEL)
                .legacyAllowAnyFileType()
                // Should be in the target configuration
                .cfg(NoTransition.createFactory()))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(module_map) -->
        Module map artifact to be used for modular builds.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("module_map", LABEL)
                .legacyAllowAnyFileType()
                // Should be in the target configuration
                .cfg(NoTransition.createFactory()))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(supports_param_files) -->
        Set to True when cc_toolchain supports using param files for linking actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("supports_param_files", BOOLEAN).value(true))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(supports_header_parsing) -->
        Set to True when cc_toolchain supports header parsing actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("supports_header_parsing", BOOLEAN).value(false))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(exec_transition_for_inputs) -->
        Set to True to build all file inputs to cc_toolchain for the exec platform,
        instead of having no transition (i.e. target platform by default).
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(CcToolchainInputsTransitionFactory.ATTR_NAME, BOOLEAN).value(true))
        // TODO(b/78578234): Make this the default and remove the late-bound versions.
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(libc_top) -->
        A collection of artifacts for libc passed as inputs to compile/linking actions.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("libc_top", LABEL)
                .allowedFileTypes()
                // Should be in the target configuration
                .cfg(NoTransition.createFactory()))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(toolchain_identifier) -->
        The identifier used to match this cc_toolchain with the corresponding
        crosstool_config.toolchain.

        <p>
          Until issue <a href="https://github.com/bazelbuild/bazel/issues/5380">#5380</a> is fixed
          this is the recommended way of associating <code>cc_toolchain</code> with
          <code>CROSSTOOL.toolchain</code>. It will be replaced by the <code>toolchain_config</code>
          attribute (<a href="https://github.com/bazelbuild/bazel/issues/5380">#5380</a>).</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("toolchain_identifier", Type.STRING)
                .nonconfigurable("Used in configuration creation")
                .value(""))
        /* <!-- #BLAZE_RULE(cc_toolchain).ATTRIBUTE(toolchain_config) -->
        The label of the rule providing <code>cc_toolchain_config_info</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr(TOOLCHAIN_CONFIG_ATTR, LABEL)
                .allowedFileTypes()
                .mandatoryProviders(CcToolchainConfigInfo.PROVIDER.id())
                .mandatory()
                // Should be in the target configuration
                .cfg(NoTransition.createFactory()))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_toolchain")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_toolchain, TYPE = OTHER, FAMILY = C / C++) -->

<p>Represents a C++ toolchain.</p>

<p>
  This rule is responsible for:

  <ul>
    <li>
      Collecting all artifacts needed for C++ actions to run. This is done by
      attributes such as <code>all_files</code>, <code>compiler_files</code>,
      <code>linker_files</code>, or other attributes ending with <code>_files</code>). These are
      most commonly filegroups globbing all required files.
    </li>
    <li>
      Generating correct command lines for C++ actions. This is done using
      <code>CcToolchainConfigInfo</code> provider (details below).
    </li>
  </ul>
</p>
<p>
  Use <code>toolchain_config</code> attribute to configure the C++ toolchain.
  See also this
  <a href="https://bazel.build/docs/cc-toolchain-config-reference">
    page
  </a> for elaborate C++ toolchain configuration and toolchain selection documentation.
</p>
<p>
  Use <code>tags = ["manual"]</code> in order to prevent toolchains from being built and configured
  unnecessarily when invoking <code>bazel build //...</code>
</p>
<!-- #END_BLAZE_RULE -->*/
