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
    name = "compile",
    // There is one mandatory positional: the Skylark rule context.
    mandatoryPositionals = 1,
    parameters = {
      @Param(
          name = "source_jars",
          positional = false,
          named = true,
          type = SkylarkList.class,
          generic1 = Artifact.class,
          defaultValue = "[]"
      ),
      @Param(
        name = "source_files",
        positional = false,
        named = true,
        type = SkylarkList.class,
        generic1 = Artifact.class,
        defaultValue = "[]"
      ),
      @Param(name = "output", positional = false, named = true, type = Artifact.class),
      @Param(
        name = "javac_opts",
        positional = false,
        named = true,
        type = SkylarkList.class,
        generic1 = String.class
      ),
      @Param(
        name = "deps",
        positional = false,
        named = true,
        type = SkylarkList.class,
        generic1 = JavaProvider.class
      ),
      @Param(
        name = "strict_deps",
        defaultValue = "OFF",
        positional = false,
        named = true,
        type = String.class
      ),
      @Param(
        name = "java_toolchain",
        positional = false,
        named = true,
        type = ConfiguredTarget.class
      ),
      @Param(
        name = "host_javabase",
        positional = false,
        named = true,
        type = ConfiguredTarget.class
      ),
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
      ConfiguredTarget hostJavabase) {
    JavaLibraryHelper helper =
        new JavaLibraryHelper(skylarkRuleContext.getRuleContext())
            .setOutput(outputJar)
            .addSourceJars(sourceJars)
            .addSourceFiles(sourceFiles)
            .setJavacOpts(javacOpts);

    List<JavaCompilationArgsProvider> compilationArgsProviders =
        JavaProvider.fetchProvidersFromList(deps, JavaCompilationArgsProvider.class);
    helper.addAllDeps(compilationArgsProviders);
    helper.setCompilationStrictDepsMode(getStrictDepsMode(strictDepsMode));
    MiddlemanProvider hostJavabaseProvider = hostJavabase.getProvider(MiddlemanProvider.class);

    NestedSet<Artifact> hostJavabaseArtifacts =
        hostJavabase == null
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
    return JavaProvider.Builder.create()
             .addProvider(
                 JavaCompilationArgsProvider.class,
                 helper.buildCompilationArgsProvider(artifacts, true))
             .addProvider(JavaSourceJarsProvider.class, createJavaSourceJarsProvider(sourceJars))
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
      SkylarkRuleContext skylarkRuleContext, String javaToolchainAttr) {
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
