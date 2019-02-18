// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import javax.annotation.Nullable;

/** Utilities for Java compilation support in Skylark. */
@SkylarkModule(name = "java_common", doc = "Utilities for Java compilation support in Starlark.")
public interface JavaCommonApi<
    FileT extends FileApi,
    JavaInfoT extends JavaInfoApi<FileT>,
    SkylarkRuleContextT extends SkylarkRuleContextApi,
    SkylarkActionFactoryT extends SkylarkActionFactoryApi> {

  @SkylarkCallable(
      name = "create_provider",
      doc =
          "This API is deprecated. It will be disabled by default in Bazel 0.23. Please use "
              + "<a href ="
              + "'https://docs.bazel.build/versions/master/skylark/lib/JavaInfo.html#JavaInfo'>"
              + "JavaInfo()</a> instead."
              + "Creates a JavaInfo from jars. compile_time/runtime_jars are the outputs of the "
              + "target providing a JavaInfo, while transitive_*_jars represent their dependencies."
              + "<p>Note: compile_time_jars and runtime_jars are not automatically merged into the "
              + "transitive jars (unless the given transitive_*_jars are empty) - if this is the "
              + "desired behaviour the user should merge the jars before creating the provider."
              + "<p>This function also creates actions to generate interface jars by default."
              + "<p>When use_ijar is True, ijar will be run on the given compile_time_jars and the "
              + "resulting interface jars will be stored as compile_jars, "
              + "while the initial jars will be stored as full_compile_jars. "
              + "<p>When use_ijar=False, the given compile_time_jars will be stored as both "
              + "compile_jars and full_compile_jars. No actions are created. "
              + "See JavaInfo#compile_jars and JavaInfo#full_compile_jars for more details."
              + "<p>Currently only "
              + "<a href='https://github.com/bazelbuild/bazel/tree/master/third_party/ijar'>"
              + "ijar</a>"
              + " is supported for generating interface jars. "
              + "Header compilation is not yet supported.",
      parameters = {
        @Param(
            name = "actions",
            type = SkylarkActionFactoryApi.class,
            noneable = true,
            defaultValue = "None",
            doc =
                "The ctx.actions object, used to register the actions for creating the "
                    + "interface jars. Only set if use_ijar=True."),
        @Param(
            name = "compile_time_jars",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = SkylarkNestedSet.class),
            },
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc = "A list or a set of jars that should be used at compilation for a given target."),
        @Param(
            name = "runtime_jars",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = SkylarkNestedSet.class),
            },
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc = "A list or a set of jars that should be used at runtime for a given target."),
        @Param(
            name = "use_ijar",
            positional = false,
            named = true,
            type = Boolean.class,
            defaultValue = "True",
            doc =
                "If True it will generate interface jars for every jar in compile_time_jars."
                    + "The generating interface jars will be stored as compile_jars "
                    + "and the initial (full) compile_time_jars will be stored as "
                    + "full_compile_jars. If False the given compile_jars will be "
                    + "stored as both compile_jars and full_compile_jars."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {
              @ParamType(type = TransitiveInfoCollectionApi.class),
              @ParamType(type = ToolchainInfoApi.class)
            },
            noneable = true,
            defaultValue = "None",
            doc =
                "A JavaToolchainInfo to be used for retrieving the ijar "
                    + "tool. Only set when use_ijar is True."),
        @Param(
            name = "transitive_compile_time_jars",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = SkylarkNestedSet.class),
            },
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc =
                "A list or set of compile time jars collected from the transitive closure of a "
                    + "rule."),
        @Param(
            name = "transitive_runtime_jars",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = SkylarkNestedSet.class),
            },
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc = "A list or set of runtime jars collected from the transitive closure of a rule."),
        @Param(
            name = "source_jars",
            positional = false,
            named = true,
            allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = SkylarkNestedSet.class),
            },
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc =
                "A list or set of output source jars that contain the uncompiled source files "
                    + "including the source files generated by annotation processors if the case.")
      },
      useLocation = true,
      useEnvironment = true)
  public JavaInfoT create(
      @Nullable Object actionsUnchecked,
      Object compileTimeJars,
      Object runtimeJars,
      Boolean useIjar,
      @Nullable Object javaToolchainUnchecked,
      Object transitiveCompileTimeJars,
      Object transitiveRuntimeJars,
      Object sourceJars,
      Location location,
      Environment environment)
      throws EvalException;

  @SkylarkCallable(
      name = "provider",
      structField = true,
      doc =
          "Returns the Java declared provider. <br>"
              + "The same value is accessible as <code>JavaInfo</code>. <br>"
              + "Prefer using <code>JavaInfo</code> in new code.")
  public ProviderApi getJavaProvider();

  @SkylarkCallable(
      name = "compile",
      doc =
          "Compiles Java source files/jars from the implementation of a Starlark rule and returns "
              + "a provider that represents the results of the compilation and can be added to "
              + "the set of providers emitted by this rule.",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "source_jars",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc =
                "A list of the jars to be compiled. At least one of source_jars or source_files"
                    + " should be specified."),
        @Param(
            name = "source_files",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc =
                "A list of the Java source files to be compiled. At least one of source_jars or "
                    + "source_files should be specified."),
        @Param(name = "output", positional = false, named = true, type = FileApi.class),
        @Param(
            name = "output_source_jar",
            positional = false,
            named = true,
            type = FileApi.class,
            noneable = true,
            defaultValue = "None",
            doc = "The output source jar. Optional. Defaults to `{output_jar}-src.jar` if unset."),
        @Param(
            name = "javac_opts",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = String.class,
            defaultValue = "[]",
            doc = "A list of the desired javac options. Optional."),
        @Param(
            name = "deps",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = JavaInfoApi.class,
            defaultValue = "[]",
            doc = "A list of dependencies. Optional."),
        @Param(
            name = "exports",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = JavaInfoApi.class,
            defaultValue = "[]",
            doc = "A list of exports. Optional."),
        @Param(
            name = "plugins",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = JavaInfoApi.class,
            defaultValue = "[]",
            doc = "A list of plugins. Optional."),
        @Param(
            name = "exported_plugins",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = JavaInfoApi.class,
            defaultValue = "[]",
            doc = "A list of exported plugins. Optional."),
        @Param(
            name = "strict_deps",
            defaultValue = "'ERROR'",
            positional = false,
            named = true,
            type = String.class,
            doc =
                "A string that specifies how to handle strict deps. Possible values: 'OFF', "
                    + "'ERROR', 'WARN' and 'DEFAULT'. For more details see "
                    + "https://docs.bazel.build/versions/master/bazel-user-manual.html#"
                    + "flag--strict_java_deps. By default 'ERROR'."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {
              @ParamType(type = TransitiveInfoCollectionApi.class),
              @ParamType(type = ToolchainInfoApi.class)
            },
            doc = "A JavaToolchainInfo to be used for this compilation. Mandatory."),
        @Param(
            name = "host_javabase",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {
              @ParamType(type = TransitiveInfoCollectionApi.class),
              @ParamType(type = JavaRuntimeInfoApi.class)
            },
            doc = "A JavaRuntimeInfo to be used for this compilation. Mandatory."),
        @Param(
            name = "sourcepath",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = FileApi.class,
            defaultValue = "[]"),
        @Param(
            name = "resources",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = FileApi.class,
            defaultValue = "[]"),
        @Param(
            name = "neverlink",
            positional = false,
            named = true,
            type = Boolean.class,
            defaultValue = "False")
      },
      useLocation = true,
      useEnvironment = true)
  public JavaInfoT createJavaCompileAction(
      SkylarkRuleContextT skylarkRuleContext,
      SkylarkList<FileT> sourceJars,
      SkylarkList<FileT> sourceFiles,
      FileT outputJar,
      Object outputSourceJar,
      SkylarkList<String> javacOpts,
      SkylarkList<JavaInfoT> deps,
      SkylarkList<JavaInfoT> exports,
      SkylarkList<JavaInfoT> plugins,
      SkylarkList<JavaInfoT> exportedPlugins,
      String strictDepsMode,
      Object javaToolchain,
      Object hostJavabase,
      SkylarkList<FileT> sourcepathEntries,
      SkylarkList<FileT> resources,
      Boolean neverlink,
      Location loc,
      Environment environment)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "run_ijar",
      doc =
          "Runs ijar on a jar, stripping it of its method bodies. This helps reduce rebuilding "
              + "of dependent jars during any recompiles consisting only of simple changes to "
              + "method implementations. The return value is typically passed to "
              + "<code><a class=\"anchor\" href=\"JavaInfo.html\">"
              + "JavaInfo</a>#compile_jar</code>.",
      parameters = {
        @Param(
            name = "actions",
            named = true,
            type = SkylarkActionFactoryApi.class,
            doc = "ctx.actions"),
        @Param(
            name = "jar",
            positional = false,
            named = true,
            type = FileApi.class,
            doc = "The jar to run ijar on."),
        @Param(
            name = "target_label",
            positional = false,
            named = true,
            type = Label.class,
            noneable = true,
            defaultValue = "None",
            doc =
                "A target label to stamp the jar with. Used for <code>add_dep</code> support. "
                    + "Typically, you would pass <code>ctx.label</code> to stamp the jar "
                    + "with the current rule's label."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {
              @ParamType(type = TransitiveInfoCollectionApi.class),
              @ParamType(type = ToolchainInfoApi.class)
            },
            doc = "A JavaToolchainInfo to used to find the ijar tool."),
      },
      useSkylarkSemantics = true,
      useLocation = true)
  public FileApi runIjar(
      SkylarkActionFactoryT actions,
      FileT jar,
      Object targetLabel,
      Object javaToolchain,
      Location location,
      StarlarkSemantics semantics)
      throws EvalException;

  @SkylarkCallable(
      name = "stamp_jar",
      doc =
          "Stamps a jar with a target label for <code>add_dep</code> support. "
              + "The return value is typically passed to "
              + "<code><a class=\"anchor\" href=\"JavaInfo.html\">"
              + "JavaInfo</a>#compile_jar</code>. "
              + "Prefer to use "
              + "<code><a class=\"anchor\" href=\"java_common.html#run_ijar\">run_ijar</a></code> "
              + "when possible.",
      parameters = {
        @Param(
            name = "actions",
            named = true,
            type = SkylarkActionFactoryApi.class,
            doc = "ctx.actions"),
        @Param(
            name = "jar",
            positional = false,
            named = true,
            type = FileApi.class,
            doc = "The jar to run stamp_jar on."),
        @Param(
            name = "target_label",
            positional = false,
            named = true,
            type = Label.class,
            doc =
                "A target label to stamp the jar with. Used for <code>add_dep</code> support. "
                    + "Typically, you would pass <code>ctx.label</code> to stamp the jar "
                    + "with the current rule's label."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {
              @ParamType(type = TransitiveInfoCollectionApi.class),
              @ParamType(type = ToolchainInfoApi.class)
            },
            doc = "A JavaToolchainInfo to used to find the stamp_jar tool."),
      },
      useSkylarkSemantics = true,
      useLocation = true)
  public FileApi stampJar(
      SkylarkActionFactoryT actions,
      FileT jar,
      Label targetLabel,
      Object javaToolchain,
      Location location,
      StarlarkSemantics semantics)
      throws EvalException;

  @SkylarkCallable(
      name = "pack_sources",
      doc =
          "Packs sources and source jars into a single source jar file. "
              + "The return value is typically passed to"
              + "<p><code><a class=\"anchor\" href=\"JavaInfo.html\">"
              + "JavaInfo</a>#source_jar</code></p>.",
      parameters = {
        @Param(
            name = "actions",
            named = true,
            type = SkylarkActionFactoryApi.class,
            doc = "ctx.actions"),
        @Param(
            name = "output_jar",
            positional = false,
            named = true,
            type = FileApi.class,
            doc = "The output jar of the rule. Used to name the resulting source jar."),
        @Param(
            name = "sources",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc = "A list of Java source files to be packed into the source jar."),
        @Param(
            name = "source_jars",
            positional = false,
            named = true,
            type = SkylarkList.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            doc = "A list of source jars to be packed into the source jar."),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {
              @ParamType(type = TransitiveInfoCollectionApi.class),
              @ParamType(type = ToolchainInfoApi.class)
            },
            doc = "A JavaToolchainInfo to used to find the ijar tool."),
        @Param(
            name = "host_javabase",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {
              @ParamType(type = TransitiveInfoCollectionApi.class),
              @ParamType(type = JavaRuntimeInfoApi.class)
            },
            doc = "A JavaRuntimeInfo to be used for packing sources."),
      },
      allowReturnNones = true,
      useSkylarkSemantics = true,
      useLocation = true)
  public FileApi packSources(
      SkylarkActionFactoryT actions,
      FileT outputJar,
      SkylarkList<FileT> sourceFiles,
      SkylarkList<FileT> sourceJars,
      Object javaToolchain,
      Object hostJavabase,
      Location location,
      StarlarkSemantics semantics)
      throws EvalException;

  @SkylarkCallable(
      name = "default_javac_opts",
      // This function is experimental for now.
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = Object.class,
            allowedTypes = {@ParamType(type = SkylarkRuleContextApi.class)},
            noneable = true,
            defaultValue = "None",
            doc = "The rule context."),
        @Param(
            name = "java_toolchain_attr",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = String.class)},
            noneable = true,
            defaultValue = "None"),
        @Param(
            name = "java_toolchain",
            positional = false,
            named = true,
            type = Object.class,
            allowedTypes = {@ParamType(type = ToolchainInfoApi.class)},
            noneable = true,
            defaultValue = "None",
            doc =
                "A JavaToolchainInfo to be used for retrieving the ijar "
                    + "tool. Only set when use_ijar is True."),
      },
      useSkylarkSemantics = true,
      useLocation = true)
  // TODO(b/78512644): migrate callers to passing explicit javacopts or using custom toolchains, and
  // delete
  public ImmutableList<String> getDefaultJavacOpts(
      Object skylarkRuleContext,
      Object javaToolchainAttr,
      Object javaToolchain,
      Location loc,
      StarlarkSemantics semantics)
      throws EvalException;

  @SkylarkCallable(
      name = "merge",
      doc = "Merges the given providers into a single JavaInfo.",
      parameters = {
        @Param(
            name = "providers",
            positional = true,
            named = false,
            type = SkylarkList.class,
            generic1 = JavaInfoApi.class,
            doc = "The list of providers to merge."),
      })
  public JavaInfoT mergeJavaProviders(SkylarkList<JavaInfoT> providers);

  @SkylarkCallable(
      name = "make_non_strict",
      doc =
          "Returns a new Java provider whose direct-jars part is the union of both the direct and"
              + " indirect jars of the given Java provider.",
      parameters = {
        @Param(
            name = "java_info",
            positional = true,
            named = false,
            type = JavaInfoApi.class,
            doc = "The java info."),
      })
  public JavaInfoT makeNonStrict(JavaInfoT javaInfo);

  @SkylarkCallable(
      name = "JavaToolchainInfo",
      doc =
          "The key used to retrieve the provider that contains information about the Java "
              + "toolchain being used.",
      structField = true)
  public ProviderApi getJavaToolchainProvider();

  @SkylarkCallable(
      name = "JavaRuntimeInfo",
      doc =
          "The key used to retrieve the provider that contains information about the Java "
              + "runtime being used.",
      structField = true)
  public ProviderApi getJavaRuntimeProvider();

  @SkylarkCallable(
      name = "is_java_toolchain_resolution_enabled_do_not_use",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
      },
      doc = "Returns true if --experimental_use_toolchain_resolution_for_java_rules is enabled.")
  boolean isJavaToolchainResolutionEnabled(SkylarkRuleContextT ruleContext) throws EvalException;
}
