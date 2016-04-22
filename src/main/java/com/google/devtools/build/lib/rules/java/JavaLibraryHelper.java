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

package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode.OFF;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.UnmodifiableIterator;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.Builder;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsStore;
import com.google.devtools.build.lib.rules.cpp.CcSpecificLinkParamsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A class to create Java compile actions in a way that is consistent with java_library. Rules that
 * generate source files and emulate java_library on top of that should use this class
 * instead of the lower-level API in JavaCompilationHelper.
 *
 * <p>Rules that want to use this class are required to have an implicit dependency on the
 * Java compiler.
 */
public final class JavaLibraryHelper {
  private static final String DEFAULT_SUFFIX_IS_EMPTY_STRING = "";

  /**
   * Function for extracting the {@link JavaCompilationArgs} - note that it also handles .jar files.
   */
  private static final Function<TransitiveInfoCollection, JavaCompilationArgsProvider>
      TO_COMPILATION_ARGS = new Function<TransitiveInfoCollection, JavaCompilationArgsProvider>() {
    @Override
    public JavaCompilationArgsProvider apply(TransitiveInfoCollection target) {
      return forTarget(target);
    }
  };

  /**
   * Contains the providers as well as the compilation outputs.
   */
  public static final class Info {
    private final Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers;
    private final JavaCompilationArtifacts compilationArtifacts;

    private Info(Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers,
        JavaCompilationArtifacts compilationArtifacts) {
      this.providers = Collections.unmodifiableMap(providers);
      this.compilationArtifacts = compilationArtifacts;
    }

    public Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> getProviders() {
      return providers;
    }

    public JavaCompilationArtifacts getCompilationArtifacts() {
      return compilationArtifacts;
    }
  }

  private final RuleContext ruleContext;
  private final BuildConfiguration configuration;
  private final String implicitAttributesSuffix;

  private Artifact output;
  private final List<Artifact> sourceJars = new ArrayList<>();
  /**
   * Contains all the dependencies; these are treated as both compile-time and runtime dependencies.
   * Some of these may not be complete configured targets; for backwards compatibility with some
   * existing code, we sometimes only have pretend dependencies that only have a single {@link
   * JavaCompilationArgsProvider}.
   */
  private final List<TransitiveInfoCollection> deps = new ArrayList<>();
  private ImmutableList<String> javacOpts = ImmutableList.of();

  private StrictDepsMode strictDepsMode = StrictDepsMode.OFF;
  private JavaClasspathMode classpathMode = JavaClasspathMode.OFF;
  private boolean emitProviders = true;
  private boolean legacyCollectCppAndJavaLinkOptions;

  public JavaLibraryHelper(RuleContext ruleContext) {
    this(ruleContext, DEFAULT_SUFFIX_IS_EMPTY_STRING);
  }

  public JavaLibraryHelper(RuleContext ruleContext, String implicitAttributesSuffix) {
    this.ruleContext = ruleContext;
    this.configuration = ruleContext.getConfiguration();
    this.classpathMode = ruleContext.getFragment(JavaConfiguration.class).getReduceJavaClasspath();
    this.implicitAttributesSuffix = implicitAttributesSuffix;
  }

  /**
   * Sets the final output jar; if this is not set, then the {@link #build} method throws an {@link
   * IllegalStateException}. Note that this class may generate not just the output itself, but also
   * a number of additional intermediate files and outputs.
   */
  public JavaLibraryHelper setOutput(Artifact output) {
    this.output = output;
    return this;
  }

  /**
   * Adds the given source jars. Any .java files in these jars will be compiled.
   */
  public JavaLibraryHelper addSourceJars(Iterable<Artifact> sourceJars) {
    Iterables.addAll(this.sourceJars, sourceJars);
    return this;
  }

  /**
   * Adds the given source jars. Any .java files in these jars will be compiled.
   */
  public JavaLibraryHelper addSourceJars(Artifact... sourceJars) {
    return this.addSourceJars(Arrays.asList(sourceJars));
  }

  /**
   * Adds the given compilation args as deps. Avoid this method, and prefer {@link #addDeps}
   * instead; this method only exists for backward compatibility and may be removed at any time.
   */
  public JavaLibraryHelper addProcessedDeps(JavaCompilationArgs... deps) {
    for (JavaCompilationArgs dep : deps) {
      this.deps.add(toTransitiveInfoCollection(dep));
    }
    return this;
  }

  private static TransitiveInfoCollection toTransitiveInfoCollection(
      final JavaCompilationArgs args) {
    return new TransitiveInfoCollection() {
      @Override
      public <P extends TransitiveInfoProvider> P getProvider(Class<P> provider) {
        if (JavaCompilationArgsProvider.class.equals(provider)) {
          return provider.cast(new JavaCompilationArgsProvider(args, args));
        }
        return null;
      }

      @Override
      public Label getLabel() {
        throw new UnsupportedOperationException();
      }

      @Override
      public BuildConfiguration getConfiguration() {
        throw new UnsupportedOperationException();
      }

      @Override
      public Object get(String providerKey) {
        throw new UnsupportedOperationException();
      }

      @Override
      public UnmodifiableIterator<TransitiveInfoProvider> iterator() {
        throw new UnsupportedOperationException();
      }
    };
  }

  /**
   * Adds the given targets as deps. These are used as both compile-time and runtime dependencies.
   */
  public JavaLibraryHelper addDeps(Iterable<? extends TransitiveInfoCollection> deps) {
    for (TransitiveInfoCollection dep : deps) {
      Preconditions.checkArgument(dep.getConfiguration() == null
          || configuration.equalsOrIsSupersetOf(dep.getConfiguration()));
      this.deps.add(dep);
    }
    return this;
  }

  /**
   * Sets the compiler options.
   */
  public JavaLibraryHelper setJavacOpts(Iterable<String> javacOpts) {
    this.javacOpts = ImmutableList.copyOf(javacOpts);
    return this;
  }

  /**
   * Sets the mode that determines how strictly dependencies are checked.
   */
  public JavaLibraryHelper setStrictDepsMode(StrictDepsMode strictDepsMode) {
    this.strictDepsMode = strictDepsMode;
    return this;
  }

  /**
   * Disables all providers, i.e., the resulting {@link Info} object will not contain any providers.
   * Avoid this method - having this class compute the providers ensures consistency among all
   * clients of this code.
   */
  public JavaLibraryHelper noProviders() {
    this.emitProviders = false;
    return this;
  }

  /**
   * Collects link options from both Java and C++ dependencies. This is never what you want, and
   * only exists for backwards compatibility.
   */
  public JavaLibraryHelper setLegacyCollectCppAndJavaLinkOptions(
      boolean legacyCollectCppAndJavaLinkOptions) {
    this.legacyCollectCppAndJavaLinkOptions = legacyCollectCppAndJavaLinkOptions;
    return this;
  }

  /**
   * Creates the compile actions and providers.
   */
  public Info build(JavaSemantics semantics) {
    Preconditions.checkState(output != null, "must have an output file; use setOutput()");
    JavaTargetAttributes.Builder attributes = new JavaTargetAttributes.Builder(semantics);
    attributes.addSourceJars(sourceJars);
    addDepsToAttributes(attributes);
    attributes.setStrictJavaDeps(strictDepsMode);
    attributes.setRuleKind(ruleContext.getRule().getRuleClass());
    attributes.setTargetLabel(ruleContext.getLabel());

    if (isStrict() && classpathMode != JavaClasspathMode.OFF) {
      JavaCompilationHelper.addDependencyArtifactsToAttributes(attributes, transformDeps());
    }

    JavaCompilationArtifacts.Builder artifactsBuilder = new JavaCompilationArtifacts.Builder();
    JavaCompilationHelper helper =
        new JavaCompilationHelper(
            ruleContext, semantics, javacOpts, attributes, implicitAttributesSuffix);
    Artifact outputDepsProto = helper.createOutputDepsProtoArtifact(output, artifactsBuilder);
    helper.createCompileAction(
        output,
        null /* manifestProtoOutput */,
        null /* gensrcOutputJar */,
        outputDepsProto,
        null /* outputMetadata */);
    helper.createCompileTimeJarAction(output, artifactsBuilder);
    artifactsBuilder.addRuntimeJar(output);
    JavaCompilationArtifacts compilationArtifacts = artifactsBuilder.build();

    Map<Class<? extends TransitiveInfoProvider>, TransitiveInfoProvider> providers =
        new LinkedHashMap<>();
    if (emitProviders) {
      providers.put(JavaCompilationArgsProvider.class,
          collectJavaCompilationArgs(compilationArtifacts));
      providers.put(JavaSourceJarsProvider.class,
          new JavaSourceJarsProvider(collectTransitiveJavaSourceJars(), sourceJars));
      providers.put(JavaRunfilesProvider.class, collectJavaRunfiles(compilationArtifacts));
      providers.put(JavaCcLinkParamsProvider.class,
          new JavaCcLinkParamsProvider(createJavaCcLinkParamsStore()));
    }
    return new Info(providers, compilationArtifacts);
  }

  private void addDepsToAttributes(JavaTargetAttributes.Builder attributes) {
    NestedSet<Artifact> directJars = null;
    if (isStrict()) {
      directJars = getNonRecursiveCompileTimeJarsFromDeps();
      if (directJars != null) {
        attributes.addDirectCompileTimeClassPathEntries(directJars);
        attributes.addDirectJars(directJars);
      }
    }

    JavaCompilationArgs args = JavaCompilationArgs.builder()
        .addTransitiveDependencies(transformDeps(), true).build();
    attributes.addCompileTimeClassPathEntries(args.getCompileTimeJars());
    attributes.addRuntimeClassPathEntries(args.getRuntimeJars());
    attributes.addInstrumentationMetadataEntries(args.getInstrumentationMetadata());
  }

  private NestedSet<Artifact> getNonRecursiveCompileTimeJarsFromDeps() {
    JavaCompilationArgs.Builder builder = JavaCompilationArgs.builder();
    builder.addTransitiveDependencies(transformDeps(), false);
    return builder.build().getCompileTimeJars();
  }

  private Iterable<JavaCompilationArgsProvider> transformDeps() {
    return Iterables.transform(deps, TO_COMPILATION_ARGS);
  }

  private static JavaCompilationArgsProvider forTarget(TransitiveInfoCollection target) {
    if (target.getProvider(JavaCompilationArgsProvider.class) != null) {
      // If the target has JavaCompilationArgs, we use those.
      return target.getProvider(JavaCompilationArgsProvider.class);
    } else {
      // Otherwise we look for any jar files. It would be good to remove this, and require
      // intermediate java_import rules in these cases.
      NestedSet<Artifact> filesToBuild =
          target.getProvider(FileProvider.class).getFilesToBuild();
      final List<Artifact> jars = new ArrayList<>();
      Iterables.addAll(jars, FileType.filter(filesToBuild, JavaSemantics.JAR));
      JavaCompilationArgs args = JavaCompilationArgs.builder()
          .addCompileTimeJars(jars)
          .addRuntimeJars(jars)
          .build();
      return new JavaCompilationArgsProvider(args, args);
    }
  }

  private boolean isStrict() {
    return strictDepsMode != OFF;
  }

  private JavaCompilationArgsProvider collectJavaCompilationArgs(
      JavaCompilationArtifacts compilationArtifacts) {
    JavaCompilationArgs javaCompilationArgs =
        collectJavaCompilationArgs(compilationArtifacts, false);
    JavaCompilationArgs recursiveJavaCompilationArgs =
        collectJavaCompilationArgs(compilationArtifacts, true);
    return new JavaCompilationArgsProvider(javaCompilationArgs, recursiveJavaCompilationArgs);
  }

  /**
   * Get compilation arguments for java compilation action.
   *
   * @param recursive a boolean specifying whether to get transitive
   *        dependencies
   * @return java compilation args
   */
  private JavaCompilationArgs collectJavaCompilationArgs(
      JavaCompilationArtifacts compilationArtifacts, boolean recursive) {
    return JavaCompilationArgs.builder()
        .merge(compilationArtifacts)
        .addTransitiveDependencies(transformDeps(), recursive)
        .build();
  }

  private NestedSet<Artifact> collectTransitiveJavaSourceJars() {
    NestedSetBuilder<Artifact> transitiveJavaSourceJarBuilder =
        NestedSetBuilder.<Artifact>stableOrder();
    transitiveJavaSourceJarBuilder.addAll(sourceJars);
    for (JavaSourceJarsProvider other : ruleContext.getPrerequisites(
        "deps", Mode.TARGET, JavaSourceJarsProvider.class)) {
      transitiveJavaSourceJarBuilder.addTransitive(other.getTransitiveSourceJars());
    }
    return transitiveJavaSourceJarBuilder.build();
  }

  private JavaRunfilesProvider collectJavaRunfiles(
      JavaCompilationArtifacts javaCompilationArtifacts) {
    Runfiles runfiles = new Runfiles.Builder(ruleContext.getWorkspaceName())
        // Compiled templates as well, for API.
        .addArtifacts(javaCompilationArtifacts.getRuntimeJars())
        .addTargets(deps, JavaRunfilesProvider.TO_RUNFILES)
        .build();
    return new JavaRunfilesProvider(runfiles);
  }

  private CcLinkParamsStore createJavaCcLinkParamsStore() {
    return new CcLinkParamsStore() {
      @Override
      protected void collect(Builder builder, boolean linkingStatically, boolean linkShared) {
        if (legacyCollectCppAndJavaLinkOptions) {
          builder.addTransitiveTargets(deps,
              JavaCcLinkParamsProvider.TO_LINK_PARAMS);
          builder.addTransitiveTargets(deps,
              CcLinkParamsProvider.TO_LINK_PARAMS,
              CcSpecificLinkParamsProvider.TO_LINK_PARAMS);
        } else {
          builder.addTransitiveTargets(deps,
              JavaCcLinkParamsProvider.TO_LINK_PARAMS,
              CcLinkParamsProvider.TO_LINK_PARAMS,
              CcSpecificLinkParamsProvider.TO_LINK_PARAMS);
        }
      }
    };
  }
}
