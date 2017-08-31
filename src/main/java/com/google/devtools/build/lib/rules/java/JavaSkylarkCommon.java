// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType.BOTH;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.List;

/** A module that contains Skylark utilities for Java support. */
@SkylarkModule(name = "java_common", doc = "Utilities for Java compilation support in Skylark.")
public class JavaSkylarkCommon {
  private final JavaSemantics javaSemantics;

  public JavaSkylarkCommon(JavaSemantics javaSemantics) {
    this.javaSemantics = javaSemantics;
  }

  @SkylarkCallable(
    name = "provider",
    structField = true,
    doc = "Returns the Java declared provider. <br>"
        + "The same value is accessible as <code>JavaInfo</code>. <br>"
        + "Prefer using <code>JavaInfo</code> in new code."
  )
  public Provider getJavaProvider() {
    return JavaInfo.PROVIDER;
  }

  @SkylarkCallable(
    name = "create_provider",
    documented = false,
    doc = "Create JavaInfo from pre-built jars. Note that compile_time_jars and "
        + "runtime_jars are not automatically merged into the recursive jars - if this is the "
        + "desired behaviour the user should merge the jars before creating the provider. "
        + "The recursive (compile/runtime) jars are the jars usually collected transitively from "
        + "dependencies.",
    parameters = {
      @Param(
        name = "compile_time_jars",
        positional = false,
        named = true,
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        defaultValue = "[]"
      ),
      @Param(
        name = "runtime_jars",
        positional = false,
        named = true,
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        defaultValue = "[]"
      ),
      @Param(
          name = "transitive_compile_time_jars",
          positional = false,
          named = true,
          allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = SkylarkNestedSet.class),
          },
          generic1 = Artifact.class,
          defaultValue = "[]"
      ),
      @Param(
          name = "transitive_runtime_jars",
          positional = false,
          named = true,
          allowedTypes = {
              @ParamType(type = SkylarkList.class),
              @ParamType(type = SkylarkNestedSet.class),
          },
          generic1 = Artifact.class,
          defaultValue = "[]"
      ),
      @Param(
        name = "source_jars",
        positional = false,
        named = true,
        allowedTypes = {
          @ParamType(type = SkylarkList.class),
          @ParamType(type = SkylarkNestedSet.class),
        },
        generic1 = Artifact.class,
        defaultValue = "[]"
      )
    }
  )
  public JavaInfo create(
      Object compileTimeJars,
      Object runtimeJars,
      Object transitiveCompileTimeJars,
      Object transitiveRuntimeJars,
      Object sourceJars)
      throws EvalException {
    NestedSet<Artifact> compileTimeJarsNestedSet = asArtifactNestedSet(compileTimeJars);
    NestedSet<Artifact> runtimeJarsNestedSet = asArtifactNestedSet(runtimeJars);
    NestedSet<Artifact> transitiveCompileTimeJarsNestedSet =
        asArtifactNestedSet(transitiveCompileTimeJars);
    NestedSet<Artifact> transitiveRuntimeJarsNestedSet = asArtifactNestedSet(transitiveRuntimeJars);

    JavaCompilationArgs javaCompilationArgs =
        JavaCompilationArgs.builder()
            .addTransitiveCompileTimeJars(compileTimeJarsNestedSet)
            .addTransitiveRuntimeJars(runtimeJarsNestedSet)
        .build();
    JavaCompilationArgs.Builder recursiveJavaCompilationArgs = JavaCompilationArgs.builder();
    if (transitiveCompileTimeJarsNestedSet.isEmpty() && transitiveRuntimeJarsNestedSet.isEmpty()) {
      recursiveJavaCompilationArgs
          .addTransitiveCompileTimeJars(compileTimeJarsNestedSet)
          .addTransitiveRuntimeJars(runtimeJarsNestedSet);
    } else {
      recursiveJavaCompilationArgs
          .addTransitiveCompileTimeJars(transitiveCompileTimeJarsNestedSet)
          .addTransitiveRuntimeJars(transitiveRuntimeJarsNestedSet);
    }

    JavaInfo javaInfo =
        JavaInfo.Builder.create()
            .addProvider(
                JavaCompilationArgsProvider.class,
                JavaCompilationArgsProvider.create(
                    javaCompilationArgs, recursiveJavaCompilationArgs.build()))
            .addProvider(
                JavaSourceJarsProvider.class,
                JavaSourceJarsProvider.create(
                    NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
                    asArtifactNestedSet(sourceJars)))
            .build();
    return javaInfo;
  }

  /**
   * Takes an Object that is either a SkylarkNestedSet or a SkylarkList of Artifacts and returns it
   * as a NestedSet.
   */
  private static NestedSet<Artifact> asArtifactNestedSet(Object o) throws EvalException {
    return o instanceof SkylarkNestedSet
        ? ((SkylarkNestedSet) o).getSet(Artifact.class)
        : NestedSetBuilder.<Artifact>naiveLinkOrder()
            .addAll(((SkylarkList<?>) o).getContents(Artifact.class, null))
            .build();
  }

  @SkylarkCallable(
    name = "compile",
    doc = "Compiles Java source files/jars from the implementation of a Skylark rule and returns a "
      + "provider that represents the results of the compilation and can be added to the set of "
      + "providers emitted by this rule.",
    // There is one mandatory positional: the Skylark rule context.
    mandatoryPositionals = 1,
    parameters = {
      @Param(
          name = "source_jars",
          positional = false,
          named = true,
          type = SkylarkList.class,
          generic1 = Artifact.class,
          defaultValue = "[]",
          doc = "A list of the jars to be compiled. At least one of source_jars or source_files"
            + " should be specified."
      ),
      @Param(
        name = "source_files",
        positional = false,
        named = true,
        type = SkylarkList.class,
        generic1 = Artifact.class,
        defaultValue = "[]",
        doc = "A list of the Java source files to be compiled. At least one of source_jars or "
          + "source_files should be specified."
      ),
      @Param(
        name = "output",
        positional = false,
        named = true,
        type = Artifact.class
      ),
      @Param(
        name = "javac_opts",
        positional = false,
        named = true,
        type = SkylarkList.class,
        generic1 = String.class,
        defaultValue =  "[]",
        doc = "A list of the desired javac options. Optional."
      ),
      @Param(
        name = "deps",
        positional = false,
        named = true,
        type = SkylarkList.class,
        generic1 = JavaInfo.class,
        defaultValue = "[]",
        doc = "A list of dependencies. Optional."
      ),
      @Param(
          name = "exports",
          positional = false,
          named = true,
          type = SkylarkList.class,
          generic1 = JavaInfo.class,
          defaultValue = "[]",
          doc = "A list of exports. Optional."
      ),
      @Param(
          name = "plugins",
          positional = false,
          named = true,
          type = SkylarkList.class,
          generic1 = JavaInfo.class,
          defaultValue = "[]",
          doc = "A list of plugins. Optional."
      ),
      @Param(
          name = "exported_plugins",
          positional = false,
          named = true,
          type = SkylarkList.class,
          generic1 = JavaInfo.class,
          defaultValue = "[]",
          doc = "A list of exported plugins. Optional."
      ),
      @Param(
        name = "strict_deps",
        defaultValue = "'ERROR'",
        positional = false,
        named = true,
        type = String.class,
        doc = "A string that specifies how to handle strict deps. Possible values: 'OFF' (silently"
          + " allowing referencing transitive dependencies) and 'ERROR' (failing to build when"
          + " transitive dependencies are used directly). By default 'OFF'."
      ),
      @Param(
        name = "java_toolchain",
        positional = false,
        named = true,
        type = ConfiguredTarget.class,
        doc = "A label pointing to a java_toolchain rule to be used for this compilation. "
          + "Mandatory."
      ),
      @Param(
        name = "host_javabase",
        positional = false,
        named = true,
        type = ConfiguredTarget.class,
        doc = "A label pointing to a JDK to be used for this compilation. Mandatory."
      ),
      @Param(
        name = "sourcepath",
        positional = false,
        named = true,
        type = SkylarkList.class,
        generic1 = Artifact.class,
        defaultValue = "[]"
      ),
      @Param(
          name = "resources",
          positional = false,
          named = true,
          type = SkylarkList.class,
          generic1 = Artifact.class,
          defaultValue = "[]"
      )
    }
  )
  public JavaInfo createJavaCompileAction(
      SkylarkRuleContext skylarkRuleContext,
      SkylarkList<Artifact> sourceJars,
      SkylarkList<Artifact> sourceFiles,
      Artifact outputJar,
      SkylarkList<String> javacOpts,
      SkylarkList<JavaInfo> deps,
      SkylarkList<JavaInfo> exports,
      SkylarkList<JavaInfo> plugins,
      SkylarkList<JavaInfo> exportedPlugins,
      String strictDepsMode,
      ConfiguredTarget javaToolchain,
      ConfiguredTarget hostJavabase,
      SkylarkList<Artifact> sourcepathEntries,
      SkylarkList<Artifact> resources) throws EvalException {

    JavaLibraryHelper helper =
        new JavaLibraryHelper(skylarkRuleContext.getRuleContext())
            .setOutput(outputJar)
            .addSourceJars(sourceJars)
            .addSourceFiles(sourceFiles)
            .addResources(resources)
            .setSourcePathEntries(sourcepathEntries)
            .setJavacOpts(javacOpts);

    List<JavaCompilationArgsProvider> depsCompilationArgsProviders =
        JavaInfo.fetchProvidersFromList(deps, JavaCompilationArgsProvider.class);
    List<JavaCompilationArgsProvider> exportsCompilationArgsProviders =
        JavaInfo.fetchProvidersFromList(exports, JavaCompilationArgsProvider.class);
    helper.addAllDeps(depsCompilationArgsProviders);
    helper.addAllExports(exportsCompilationArgsProviders);
    helper.setCompilationStrictDepsMode(getStrictDepsMode(strictDepsMode));
    MiddlemanProvider hostJavabaseProvider = hostJavabase.getProvider(MiddlemanProvider.class);

    helper.addAllPlugins(
        JavaInfo.fetchProvidersFromList(plugins, JavaPluginInfoProvider.class));
    helper.addAllPlugins(JavaInfo.fetchProvidersFromList(deps, JavaPluginInfoProvider.class));

    NestedSet<Artifact> hostJavabaseArtifacts =
        hostJavabaseProvider == null
            ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
            : hostJavabaseProvider.getMiddlemanArtifact();
    JavaToolchainProvider javaToolchainProvider =
        javaToolchain.getProvider(JavaToolchainProvider.class);
    if (javaToolchain == null) {
      throw new EvalException(null, javaToolchain.getLabel() + " is not a java_toolchain rule.");
    }
    JavaCompilationArtifacts artifacts =
        helper.build(
            javaSemantics,
            javaToolchainProvider,
            hostJavabaseArtifacts,
            SkylarkList.createImmutable(ImmutableList.<Artifact>of()));
    JavaRuleOutputJarsProvider javaRuleOutputJarsProvider =
        JavaRuleOutputJarsProvider.builder().addOutputJar(
            new JavaRuleOutputJarsProvider.OutputJar(outputJar, /* ijar */ null, sourceJars))
        .build();
    JavaCompilationArgsProvider javaCompilationArgsProvider =
        helper.buildCompilationArgsProvider(artifacts, true);
    Runfiles runfiles = new Runfiles.Builder(skylarkRuleContext.getWorkspaceName()).addArtifacts(
        javaCompilationArgsProvider.getRecursiveJavaCompilationArgs().getRuntimeJars()).build();

    JavaPluginInfoProvider transitivePluginsProvider =
        JavaPluginInfoProvider.merge(Iterables.concat(
          JavaInfo.getProvidersFromListOfJavaProviders(
              JavaPluginInfoProvider.class, exportedPlugins),
          JavaInfo.getProvidersFromListOfJavaProviders(
              JavaPluginInfoProvider.class, exports)
        ));

    return JavaInfo.Builder.create()
             .addProvider(JavaCompilationArgsProvider.class, javaCompilationArgsProvider)
             .addProvider(JavaSourceJarsProvider.class, createJavaSourceJarsProvider(sourceJars))
             .addProvider(JavaRuleOutputJarsProvider.class, javaRuleOutputJarsProvider)
             .addProvider(JavaRunfilesProvider.class, new JavaRunfilesProvider(runfiles))
             .addProvider(JavaPluginInfoProvider.class, transitivePluginsProvider)
             .build();
  }

  /**
   * Creates a {@link JavaSourceJarsProvider} from the given list of source jars.
   */
  private static JavaSourceJarsProvider createJavaSourceJarsProvider(List<Artifact> sourceJars) {
    NestedSet<Artifact> javaSourceJars =
        NestedSetBuilder.<Artifact>stableOrder().addAll(sourceJars).build();
    return JavaSourceJarsProvider.create(
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER), javaSourceJars);
  }

  @SkylarkCallable(
      name = "default_javac_opts",
      // This function is experimental for now.
      documented = false,
      // There's only one mandatory positional,the Skylark context
      mandatoryPositionals = 1,
      parameters = {
        @Param(name = "java_toolchain_attr", positional = false, named = true, type = String.class)
      }
  )
  public static ImmutableList<String> getDefaultJavacOpts(
      SkylarkRuleContext skylarkRuleContext, String javaToolchainAttr) throws EvalException {
    RuleContext ruleContext = skylarkRuleContext.getRuleContext();
    ConfiguredTarget javaToolchainConfigTarget =
        (ConfiguredTarget) skylarkRuleContext.getAttr().getValue(javaToolchainAttr);
    JavaToolchainProvider toolchain =
        javaToolchainConfigTarget.getProvider(JavaToolchainProvider.class);
    if (toolchain == null) {
      throw new EvalException(null, javaToolchainAttr + " is not a java_toolchain rule label");
    }
    return ImmutableList.copyOf(Iterables.concat(
        toolchain.getJavacOptions(), ruleContext.getTokenizedStringListAttr("javacopts")));
  }

  @SkylarkCallable(
    name = "merge",
    doc = "Merges the given providers into a single JavaInfo.",
    // We have one positional argument: the list of providers to merge.
    mandatoryPositionals = 1
  )
  public static JavaInfo mergeJavaProviders(SkylarkList<JavaInfo> providers) {
    return JavaInfo.merge(providers);
  }

  // TODO(b/65113771): Remove this method because it's incorrect.
  @SkylarkCallable(
    name = "make_non_strict",
    doc =
        "Returns a new Java provider whose direct-jars part is the union of both the direct and"
            + " indirect jars of the given Java provider.",
    // There's only one mandatory positional, the Java provider.
    mandatoryPositionals = 1
  )
  public static JavaInfo makeNonStrict(JavaInfo javaInfo) {
    JavaCompilationArgsProvider directCompilationArgs =
        makeNonStrict(javaInfo.getProvider(JavaCompilationArgsProvider.class));

    return JavaInfo.Builder.copyOf(javaInfo)
        // Overwrites the old provider.
        .addProvider(JavaCompilationArgsProvider.class, directCompilationArgs)
        .build();
  }

  /**
   * Returns a new JavaCompilationArgsProvider whose direct-jars part is the union of both the
   * direct and indirect jars of 'provider'.
   */
  private static JavaCompilationArgsProvider makeNonStrict(JavaCompilationArgsProvider provider) {
    JavaCompilationArgs.Builder directCompilationArgs = JavaCompilationArgs.builder();
    directCompilationArgs
        .addTransitiveArgs(provider.getJavaCompilationArgs(), BOTH)
        .addTransitiveArgs(provider.getRecursiveJavaCompilationArgs(), BOTH);
    return JavaCompilationArgsProvider.create(
        directCompilationArgs.build(),
        provider.getRecursiveJavaCompilationArgs(),
        provider.getCompileTimeJavaDependencyArtifacts(),
        provider.getRunTimeJavaDependencyArtifacts());
  }

  private static StrictDepsMode getStrictDepsMode(String strictDepsMode) {
    switch (strictDepsMode) {
      case "OFF":
        return StrictDepsMode.OFF;
      case "ERROR":
        return StrictDepsMode.ERROR;
      default:
        throw new IllegalArgumentException(
            "StrictDepsMode "
                + strictDepsMode
                + " not allowed."
                + " Only OFF and ERROR values are accepted.");
    }
  }

  @SkylarkCallable(
    name = JavaRuntimeInfo.SKYLARK_NAME,
    doc =
        "The key used to retrieve the provider that contains information about the Java "
            + "runtime being used.",
    structField = true
  )
  public static Provider getJavaRuntimeProvider() {
    return JavaRuntimeInfo.PROVIDER;
  }
}
