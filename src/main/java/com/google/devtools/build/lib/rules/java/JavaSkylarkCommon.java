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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.rules.java.proto.StrictDepsUtils;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
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
    doc = "Returns the Java declared provider."
  )
  public ClassObjectConstructor getJavaProvider() {
    return JavaProvider.JAVA_PROVIDER;
  }

  @SkylarkCallable(
      name = "create_provider",
      documented = false,
      parameters = {
          @Param(
              name = "compile_time_jars",
              positional = false,
              named = true,
              type = SkylarkList.class,
              generic1 = Artifact.class,
              defaultValue = "[]"
          ),
          @Param(
              name = "runtime_jars",
              positional = false,
              named = true,
              type = SkylarkList.class,
              generic1 = Artifact.class,
              defaultValue = "[]"
          )
      }
  )
  public JavaProvider create(
      SkylarkList<Artifact> compileTimeJars,
      SkylarkList<Artifact> runtimeJars) {
    JavaCompilationArgs javaCompilationArgs = JavaCompilationArgs.builder()
        .addCompileTimeJars(compileTimeJars)
        .addRuntimeJars(runtimeJars)
        .build();
    JavaCompilationArgs recursiveJavaCompilationArgs = JavaCompilationArgs.builder()
        .addCompileTimeJars(compileTimeJars)
        .addRuntimeJars(runtimeJars).build();
    JavaProvider javaProvider = JavaProvider.Builder.create().addProvider(
              JavaCompilationArgsProvider.class,
              JavaCompilationArgsProvider.create(javaCompilationArgs, recursiveJavaCompilationArgs))
          .build();
    return javaProvider;
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
      @Param(name = "output", positional = false, named = true, type = Artifact.class),
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
        generic1 = JavaProvider.class,
        defaultValue = "[]",
        doc = "A list of dependencies. Optional."
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
      )
    }
  )
  public JavaProvider createJavaCompileAction(
      SkylarkRuleContext skylarkRuleContext,
      SkylarkList<Artifact> sourceJars,
      SkylarkList<Artifact> sourceFiles,
      Artifact outputJar,
      SkylarkList<String> javacOpts,
      SkylarkList<JavaProvider> deps,
      String strictDepsMode,
      ConfiguredTarget javaToolchain,
      ConfiguredTarget hostJavabase,
      SkylarkList<Artifact> sourcepathEntries) throws EvalException {

    JavaLibraryHelper helper =
        new JavaLibraryHelper(skylarkRuleContext.getRuleContext())
            .setOutput(outputJar)
            .addSourceJars(sourceJars)
            .addSourceFiles(sourceFiles)
            .setSourcePathEntries(sourcepathEntries)
            .setJavacOpts(javacOpts);

    List<JavaCompilationArgsProvider> compilationArgsProviders =
        JavaProvider.fetchProvidersFromList(deps, JavaCompilationArgsProvider.class);
    helper.addAllDeps(compilationArgsProviders);
    helper.setCompilationStrictDepsMode(getStrictDepsMode(strictDepsMode));
    MiddlemanProvider hostJavabaseProvider = hostJavabase.getProvider(MiddlemanProvider.class);

    NestedSet<Artifact> hostJavabaseArtifacts =
        hostJavabaseProvider == null
            ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
            : hostJavabaseProvider.getMiddlemanArtifact();
    JavaToolchainProvider javaToolchainProvider =
        checkNotNull(javaToolchain.getProvider(JavaToolchainProvider.class));
    JavaCompilationArgs artifacts =
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
    return JavaProvider.Builder.create()
             .addProvider(JavaCompilationArgsProvider.class, javaCompilationArgsProvider)
             .addProvider(JavaSourceJarsProvider.class, createJavaSourceJarsProvider(sourceJars))
             .addProvider(JavaRuleOutputJarsProvider.class, javaRuleOutputJarsProvider)
             .addProvider(JavaRunfilesProvider.class, new JavaRunfilesProvider(runfiles))
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
  public static List<String> getDefaultJavacOpts(
      SkylarkRuleContext skylarkRuleContext, String javaToolchainAttr) throws EvalException {
    RuleContext ruleContext = skylarkRuleContext.getRuleContext();
    ConfiguredTarget javaToolchainConfigTarget =
        (ConfiguredTarget) checkNotNull(skylarkRuleContext.getAttr().getValue(javaToolchainAttr));
    JavaToolchainProvider toolchain =
        checkNotNull(javaToolchainConfigTarget.getProvider(JavaToolchainProvider.class));
    return ImmutableList.copyOf(Iterables.concat(
        toolchain.getJavacOptions(), ruleContext.getTokenizedStringListAttr("javacopts")));
  }

  @SkylarkCallable(
    name = "merge",
    doc = "Merges the given providers into a single java_common.provider.",
    // We have one positional argument: the list of providers to merge.
    mandatoryPositionals = 1
  )
  public static JavaProvider mergeJavaProviders(SkylarkList<JavaProvider> providers) {
    return JavaProvider.merge(providers);
  }

  @SkylarkCallable(
      name = "make_non_strict",
      doc = "Returns a new Java provider whose direct-jars part is the union of both the direct and"
          + " indirect jars of the given Java provider.",
      // There's only one mandatory positional, the Java provider.
      mandatoryPositionals = 1
  )
  public static JavaProvider makeNonStrict(JavaProvider javaProvider) {
    JavaCompilationArgsProvider directCompilationArgs =
        StrictDepsUtils.makeNonStrict(javaProvider.getProvider(JavaCompilationArgsProvider.class));

    return JavaProvider.Builder.copyOf(javaProvider)
        // Overwrites the old provider.
        .addProvider(JavaCompilationArgsProvider.class, directCompilationArgs)
        .build();
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
}
