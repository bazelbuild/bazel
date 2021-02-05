// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST_DICT;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;

/** Rule definition for {@code java_toolchain} */
public final class JavaToolchainRule<C extends JavaToolchain> implements RuleDefinition {

  private final Class<C> ruleClass;

  public static <C extends JavaToolchain> JavaToolchainRule<C> create(Class<C> ruleClass) {
    return new JavaToolchainRule<C>(ruleClass);
  }

  private JavaToolchainRule(Class<C> ruleClass) {
    this.ruleClass = ruleClass;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(JavaConfiguration.class)
        /* <!-- #BLAZE_RULE(java_plugin).ATTRIBUTE(output_licenses) -->
        See <a href="${link common-definitions#binary.output_licenses}"><code>common attributes
        </code></a>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("output_licenses", LICENSE))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(source_version) -->
        The Java source version (e.g., '6' or '7'). It specifies which set of code structures
        are allowed in the Java source code.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("source_version", STRING)) // javac -source flag value.
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(target_version) -->
        The Java target version (e.g., '6' or '7'). It specifies for which Java runtime the class
        should be build.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("target_version", STRING)) // javac -target flag value.
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(bootclasspath) -->
        The Java target bootclasspath entries. Corresponds to javac's -bootclasspath flag.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("bootclasspath", LABEL_LIST)
                .value(ImmutableList.of())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                // This should be in the target configuration.
                .cfg(NoTransition.createFactory()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(xlint) -->
        The list of warning to add or removes from default list. Precedes it with a dash to
        removes it. Please see the Javac documentation on the -Xlint options for more information.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("xlint", STRING_LIST).value(ImmutableList.<String>of()))
        .add(
            attr("misc", STRING_LIST)
                .undocumented("use javacopts instead")
                .value(ImmutableList.<String>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(javacopts) -->
        The list of extra arguments for the Java compiler. Please refer to the Java compiler
        documentation for the extensive list of possible Java compiler flags.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("javacopts", STRING_LIST).value(ImmutableList.<String>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(jvm_opts) -->
        The list of arguments for the JVM when invoking the Java compiler. Please refer to the Java
        virtual machine documentation for the extensive list of possible flags for this option.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("jvm_opts", STRING_LIST).value(ImmutableList.<String>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(javabuilder_jvm_opts) -->
        The list of arguments for the JVM when invoking JavaBuilder.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("javabuilder_jvm_opts", STRING_LIST).value(ImmutableList.<String>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(javabuilder_data) -->
        Labels of data available for label-expansion in javabuilder_jvm_opts.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("javabuilder_data", LABEL_LIST)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(turbine_jvm_opts) -->
        The list of arguments for the JVM when invoking turbine.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("turbine_jvm_opts", STRING_LIST).value(ImmutableList.<String>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(turbine_data) -->
        Labels of data available for label-expansion in turbine_jvm_opts.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("turbine_data", LABEL_LIST)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(javac_supports_workers) -->
        True if JavaBuilder supports running as a persistent worker, false if it doesn't.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("javac_supports_workers", BOOLEAN).value(true))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(javac_supports_multiplex_workers) -->
        True if JavaBuilder supports running as a multiplex persistent worker, false if it doesn't.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("javac_supports_multiplex_workers", BOOLEAN).value(true))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(tools) -->
        Labels of tools available for label-expansion in jvm_opts.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("tools", LABEL_LIST)
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(javabuilder) -->
        Label of the JavaBuilder deploy jar.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("javabuilder", LABEL_LIST)
                .mandatory()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(singlejar) -->
        Label of the SingleJar deploy jar.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("singlejar", LABEL_LIST)
                .mandatory()
                .singleArtifact()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(genclass) -->
        Label of the GenClass deploy jar.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("genclass", LABEL_LIST)
                .mandatory()
                .singleArtifact()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(resourcejar) -->
        Label of the resource jar builder executable.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("resourcejar", LABEL_LIST)
                .singleArtifact()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(timezone_data) -->
        Label of a resource jar containing timezone data. If set, the timezone data is added as an
        implicitly runtime dependency of all java_binary rules.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("timezone_data", LABEL)
                .singleArtifact()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(ijar) -->
        Label of the ijar executable.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("ijar", LABEL_LIST)
                .mandatory()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(header_compiler) -->
        Label of the header compiler. Required if --java_header_compilation is enabled.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("header_compiler", LABEL_LIST)
                .singleArtifact()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(header_compiler_direct) -->
        Optional label of the header compiler to use for direct classpath actions that do not
        include any API-generating annotation processors.

        <p>This tool does not support annotation processing.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("header_compiler_direct", LABEL_LIST)
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        .add(
            attr("header_compiler_builtin_processors", STRING_LIST)
                .undocumented("internal")
                .value(ImmutableList.<String>of()))
        .add(
            attr("reduced_classpath_incompatible_processors", STRING_LIST)
                .undocumented("internal")
                .value(ImmutableList.<String>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(oneversion) -->
        Label of the one-version enforcement binary.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("oneversion", LABEL)
                .singleArtifact()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(oneversion_whitelist) -->
        Label of the one-version whitelist.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("oneversion_whitelist", LABEL)
                .singleArtifact()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(forcibly_disable_header_compilation) -->
        Overrides --java_header_compilation to disable header compilation on platforms that do not
        support it, e.g. JDK 7 Bazel.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("forcibly_disable_header_compilation", BOOLEAN).value(false))
        .add(
            attr("compatible_javacopts", STRING_LIST_DICT)
                .undocumented("internal")
                .value(ImmutableMap.<String, List<String>>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(package_configuration) -->
        Configuration that should be applied to the specified package groups.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("package_configuration", LABEL_LIST)
                .allowedFileTypes()
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .mandatoryBuiltinProviders(
                    ImmutableList.of(JavaPackageConfigurationProvider.class)))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(jacocorunner) -->
        Label of the JacocoCoverageRunner deploy jar.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("jacocorunner", LABEL)
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(proguard_allowlister) -->
        Label of the Proguard allowlister.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("proguard_allowlister", LABEL)
                // This needs to be in the execution configuration.
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                // TODO(b/170769708): set explicitly in Bazel and remove this default
                .value(env.getToolsLabel("//tools/jdk:proguard_whitelister"))
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(java_runtime) -->
        The java_runtime to use with this toolchain. It defaults to java_runtime
        in execution configuration.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("java_runtime", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .mandatory()
                .mandatoryProviders(ToolchainInfo.PROVIDER.id())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .useOutputLicenses())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(android_lint_runner) -->
        Label of the Android Lint runner, if any.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("android_lint_runner", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .exec())
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(android_lint_opts) -->
        The list of Android Lint arguments.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("android_lint_opts", STRING_LIST).value(ImmutableList.<String>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(android_lint_data) -->
        Labels of tools available for label-expansion in android_lint_jvm_opts.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("android_lint_data", LABEL_LIST)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(FileTypeSet.ANY_FILE))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(android_lint_jvm_opts) -->
        The list of arguments for the JVM when invoking Android Lint.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("android_lint_jvm_opts", STRING_LIST).value(ImmutableList.<String>of()))
        /* <!-- #BLAZE_RULE(java_toolchain).ATTRIBUTE(android_lint_package_configuration) -->
        Android Lint Configuration that should be applied to the specified package groups.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("android_lint_package_configuration", LABEL_LIST)
                .allowedFileTypes()
                .cfg(ExecutionTransitionFactory.create())
                .mandatoryBuiltinProviders(
                    ImmutableList.of(JavaPackageConfigurationProvider.class)))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_toolchain")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(ruleClass)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = java_toolchain, TYPE = OTHER, FAMILY = Java) -->

<p>
Specifies the configuration for the Java compiler. Which toolchain to be used can be changed through
the --java_toolchain argument. Normally you should not write those kind of rules unless you want to
tune your Java compiler.
</p>

<h4>Examples</h4>

<p>A simple example would be:
</p>

<pre class="code">
java_toolchain(
    name = "toolchain",
    source_version = "7",
    target_version = "7",
    bootclasspath = ["//tools/jdk:bootclasspath"],
    xlint = [ "classfile", "divzero", "empty", "options", "path" ],
    javacopts = [ "-g" ],
    javabuilder = ":JavaBuilder_deploy.jar",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
