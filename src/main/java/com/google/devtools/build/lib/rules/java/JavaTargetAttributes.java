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

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * An object that captures the temporary state we need to pass around while the initialization hook
 * for a java rule is running.
 */
public class JavaTargetAttributes {

  private static void checkJar(Artifact classPathEntry) {
    if (!JavaSemantics.JAR.matches(classPathEntry.getFilename())) {
      throw new IllegalArgumentException("not a jar file: " + classPathEntry.prettyPrint());
    }
  }

  /** A builder class for JavaTargetAttributes. */
  public static class Builder {

    // The order of source files is important, and there must not be duplicates.
    // Unfortunately, there is no interface in Java that represents a collection
    // without duplicates that has a stable and deterministic iteration order,
    // but is not sorted according to a property of the elements. Thus we are
    // stuck with Set.
    private final List<Artifact> sourceFiles = new ArrayList<>();

    private final NestedSetBuilder<Artifact> runtimeClassPath = NestedSetBuilder.naiveLinkOrder();

    private final NestedSetBuilder<Artifact> compileTimeClassPathBuilder =
        NestedSetBuilder.naiveLinkOrder();

    private NestedSet<Artifact> bootClassPath = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private ImmutableList<Artifact> sourcePath = ImmutableList.of();
    private final ImmutableList.Builder<Artifact> nativeLibraries = ImmutableList.builder();

    private JavaPluginInfoProvider plugins = JavaPluginInfoProvider.empty();

    private final Map<PathFragment, Artifact> resources = new LinkedHashMap<>();
    private final NestedSetBuilder<Artifact> resourceJars = NestedSetBuilder.stableOrder();
    private final ImmutableList.Builder<Artifact> messages = ImmutableList.builder();
    private final List<Artifact> sourceJars = new ArrayList<>();

    private final ImmutableList.Builder<Artifact> classPathResources = ImmutableList.builder();

    private final ImmutableSet.Builder<Artifact> additionalOutputs = ImmutableSet.builder();

    /** @see {@link #setStrictJavaDeps}. */
    private StrictDepsMode strictJavaDeps = StrictDepsMode.ERROR;

    private final NestedSetBuilder<Artifact> directJarsBuilder = NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> compileTimeDependencyArtifacts =
        NestedSetBuilder.stableOrder();
    private Label targetLabel;
    @Nullable private String injectingRuleKind;

    private final NestedSetBuilder<Artifact> excludedArtifacts = NestedSetBuilder.naiveLinkOrder();

    private boolean built = false;

    private final JavaSemantics semantics;

    public Builder(JavaSemantics semantics) {
      this.semantics = semantics;
    }

    public Builder addSourceArtifacts(Iterable<Artifact> sourceArtifacts) {
      Preconditions.checkArgument(!built);
      for (Artifact srcArtifact : sourceArtifacts) {
        String srcFilename = srcArtifact.getExecPathString();
        if (JavaSemantics.SOURCE_JAR.matches(srcFilename)) {
          sourceJars.add(srcArtifact);
        } else if (JavaSemantics.PROPERTIES.matches(srcFilename)) {
          // output files of the message compiler
          resources.put(
              semantics.getDefaultJavaResourcePath(srcArtifact.getRootRelativePath()), srcArtifact);
        } else if (JavaSemantics.JAVA_SOURCE.matches(srcFilename)) {
          sourceFiles.add(srcArtifact);
        } else {
          // try specific cases from the semantics.
          semantics.addArtifactToJavaTargetAttribute(this, srcArtifact);
        }
      }
      return this;
    }

    public Builder addSourceFiles(Iterable<Artifact> sourceFiles) {
      Preconditions.checkArgument(!built);
      for (Artifact artifact : sourceFiles) {
        if (JavaSemantics.JAVA_SOURCE.matches(artifact.getFilename())) {
          this.sourceFiles.add(artifact);
        }
      }
      return this;
    }

    public Builder merge(JavaCompilationArgsProvider context) {
      Preconditions.checkArgument(!built);
      addCompileTimeClassPathEntries(context.getTransitiveCompileTimeJars());
      addRuntimeClassPathEntries(context.getRuntimeJars());
      return this;
    }

    public Builder addSourceJars(Collection<Artifact> sourceJars) {
      Preconditions.checkArgument(!built);
      this.sourceJars.addAll(sourceJars);
      return this;
    }

    public Builder addSourceJar(Artifact sourceJar) {
      Preconditions.checkArgument(!built);
      this.sourceJars.add(sourceJar);
      return this;
    }

    public Builder addRuntimeClassPathEntry(Artifact classPathEntry) {
      Preconditions.checkArgument(!built);
      checkJar(classPathEntry);
      runtimeClassPath.add(classPathEntry);
      return this;
    }

    public Builder addRuntimeClassPathEntries(NestedSet<Artifact> classPathEntries) {
      Preconditions.checkArgument(!built);
      runtimeClassPath.addTransitive(classPathEntries);
      return this;
    }

    public Builder addCompileTimeClassPathEntry(Artifact entry) {
      Preconditions.checkArgument(!built);
      compileTimeClassPathBuilder.add(entry);
      return this;
    }

    public Builder addCompileTimeClassPathEntries(NestedSet<Artifact> entries) {
      Preconditions.checkArgument(!built);
      compileTimeClassPathBuilder.addTransitive(entries);
      return this;
    }

    public Builder setTargetLabel(Label targetLabel) {
      Preconditions.checkArgument(!built);
      this.targetLabel = targetLabel;
      return this;
    }

    public Builder setInjectingRuleKind(@Nullable String injectingRuleKind) {
      Preconditions.checkArgument(!built);
      this.injectingRuleKind = injectingRuleKind;
      return this;
    }

    /**
     * Sets the bootclasspath to be passed to the Java compiler.
     *
     * <p>If this method is called, then the bootclasspath specified in this JavaTargetAttributes
     * instance overrides the default bootclasspath.
     */
    public Builder setBootClassPath(NestedSet<Artifact> jars) {
      Preconditions.checkArgument(!built);
      Preconditions.checkArgument(!jars.isEmpty());
      Preconditions.checkState(bootClassPath.isEmpty());
      bootClassPath = jars;
      return this;
    }

    /** Sets the sourcepath to be passed to the Java compiler. */
    public Builder setSourcePath(ImmutableList<Artifact> artifacts) {
      Preconditions.checkArgument(!built);
      Preconditions.checkArgument(sourcePath.isEmpty());
      this.sourcePath = artifacts;
      return this;
    }

    public Builder addExcludedArtifacts(NestedSet<Artifact> toExclude) {
      Preconditions.checkArgument(!built);
      excludedArtifacts.addTransitive(toExclude);
      return this;
    }

    /**
     * Controls how strict the javac compiler will be in checking correct use of direct
     * dependencies.
     *
     * <p>Defaults to {@link StrictDepsMode#ERROR}.
     *
     * @param strictDeps one of WARN, ERROR or OFF
     */
    public Builder setStrictJavaDeps(StrictDepsMode strictDeps) {
      Preconditions.checkArgument(!built);
      strictJavaDeps = strictDeps;
      return this;
    }

    /**
     * In tandem with strictJavaDeps, directJars represents a subset of the compile-time classpath
     * jars that were provided by direct dependencies. When strictJavaDeps is OFF, there is no need
     * to provide directJars, and no extra information is passed to javac. When strictJavaDeps is
     * set to WARN or ERROR, the compiler command line will include extra flags to indicate the
     * warning/error policy and to map the classpath jars to direct or transitive dependencies,
     * using the information in directJars. The compiler command line will include an extra flag to
     * indicate which classpath jars are direct dependencies.
     */
    public Builder addDirectJars(NestedSet<Artifact> directJars) {
      Preconditions.checkArgument(!built);
      this.directJarsBuilder.addTransitive(directJars);
      return this;
    }

    public Builder addDirectJar(Artifact directJar) {
      Preconditions.checkArgument(!built);
      this.directJarsBuilder.add(directJar);
      return this;
    }

    public Builder addCompileTimeDependencyArtifacts(NestedSet<Artifact> dependencyArtifacts) {
      Preconditions.checkArgument(!built);
      compileTimeDependencyArtifacts.addTransitive(dependencyArtifacts);
      return this;
    }

    public Builder addNativeLibrary(Artifact nativeLibrary) {
      Preconditions.checkArgument(!built);
      String name = nativeLibrary.getFilename();
      if (CppFileTypes.INTERFACE_SHARED_LIBRARY.matches(name)) {
        return this;
      }
      if (!(CppFileTypes.SHARED_LIBRARY.matches(name)
          || CppFileTypes.VERSIONED_SHARED_LIBRARY.matches(name))) {
        throw new IllegalArgumentException("not a shared library :" + nativeLibrary.prettyPrint());
      }
      nativeLibraries.add(nativeLibrary);
      return this;
    }

    public Builder addNativeLibraries(Iterable<Artifact> nativeLibraries) {
      Preconditions.checkArgument(!built);
      for (Artifact nativeLibrary : nativeLibraries) {
        addNativeLibrary(nativeLibrary);
      }
      return this;
    }

    public Builder addMessages(Collection<Artifact> messages) {
      Preconditions.checkArgument(!built);
      this.messages.addAll(messages);
      return this;
    }

    public Builder addMessage(Artifact messagesArtifact) {
      Preconditions.checkArgument(!built);
      this.messages.add(messagesArtifact);
      return this;
    }

    public Builder addResource(PathFragment execPath, Artifact resource) {
      Preconditions.checkArgument(!built);
      this.resources.put(execPath, resource);
      return this;
    }

    public Builder addResourceJars(NestedSet<Artifact> resourceJars) {
      Preconditions.checkArgument(!built);
      this.resourceJars.addTransitive(resourceJars);
      return this;
    }

    public Builder addPlugin(JavaPluginInfoProvider plugins) {
      Preconditions.checkArgument(!built);
      this.plugins = JavaPluginInfoProvider.merge(this.plugins, plugins);
      return this;
    }

    public Builder addClassPathResources(List<Artifact> classPathResources) {
      Preconditions.checkArgument(!built);
      this.classPathResources.addAll(classPathResources);
      return this;
    }

    public Builder addClassPathResource(Artifact classPathResource) {
      Preconditions.checkArgument(!built);
      this.classPathResources.add(classPathResource);
      return this;
    }

    /** Adds additional outputs to this target's compile action. */
    public Builder addAdditionalOutputs(Iterable<Artifact> outputs) {
      Preconditions.checkArgument(!built);
      additionalOutputs.addAll(outputs);
      return this;
    }

    public JavaTargetAttributes build() {
      built = true;
      NestedSet<Artifact> directJars = directJarsBuilder.build();
      NestedSet<Artifact> compileTimeClassPath =
          NestedSetBuilder.<Artifact>naiveLinkOrder()
              .addTransitive(directJars)
              .addTransitive(compileTimeClassPathBuilder.build())
              .build();
      return new JavaTargetAttributes(
          ImmutableSet.copyOf(sourceFiles),
          runtimeClassPath.build(),
          compileTimeClassPath,
          bootClassPath,
          sourcePath,
          nativeLibraries.build(),
          plugins,
          ImmutableMap.copyOf(resources),
          resourceJars.build(),
          messages.build(),
          ImmutableList.copyOf(sourceJars),
          classPathResources.build(),
          additionalOutputs.build(),
          directJars,
          compileTimeDependencyArtifacts.build(),
          targetLabel,
          injectingRuleKind,
          excludedArtifacts.build(),
          strictJavaDeps);
    }

    // TODO(bazel-team): delete the following method - users should use the built
    // JavaTargetAttributes instead of accessing mutable state in the Builder.
    /** @deprecated prefer {@link JavaTargetAttributes#hasSources} */
    @Deprecated
    public boolean hasSources() {
      return !sourceFiles.isEmpty() || !sourceJars.isEmpty();
    }

    /** @deprecated prefer {@link JavaTargetAttributes#getSourceFiles} */
    @Deprecated
    public boolean hasSourceFiles() {
      return !sourceFiles.isEmpty();
    }
  }

  //
  // -------------------------- END OF BUILDER CLASS -------------------------
  //

  private final ImmutableSet<Artifact> sourceFiles;

  private final NestedSet<Artifact> runtimeClassPath;
  private final NestedSet<Artifact> compileTimeClassPath;

  private final NestedSet<Artifact> bootClassPath;
  private final ImmutableList<Artifact> sourcePath;
  private final ImmutableList<Artifact> nativeLibraries;

  private final JavaPluginInfoProvider plugins;

  private final ImmutableMap<PathFragment, Artifact> resources;
  private final NestedSet<Artifact> resourceJars;

  private final ImmutableList<Artifact> messages;
  private final ImmutableList<Artifact> sourceJars;

  private final ImmutableList<Artifact> classPathResources;

  private final ImmutableSet<Artifact> additionalOutputs;

  private final NestedSet<Artifact> directJars;
  private final NestedSet<Artifact> compileTimeDependencyArtifacts;
  private final Label targetLabel;
  @Nullable private final String injectingRuleKind;

  private final NestedSet<Artifact> excludedArtifacts;
  private final StrictDepsMode strictJavaDeps;

  /** Constructor of JavaTargetAttributes. */
  private JavaTargetAttributes(
      ImmutableSet<Artifact> sourceFiles,
      NestedSet<Artifact> runtimeClassPath,
      NestedSet<Artifact> compileTimeClassPath,
      NestedSet<Artifact> bootClassPath,
      ImmutableList<Artifact> sourcePath,
      ImmutableList<Artifact> nativeLibraries,
      JavaPluginInfoProvider plugins,
      ImmutableMap<PathFragment, Artifact> resources,
      NestedSet<Artifact> resourceJars,
      ImmutableList<Artifact> messages,
      ImmutableList<Artifact> sourceJars,
      ImmutableList<Artifact> classPathResources,
      ImmutableSet<Artifact> additionalOutputs,
      NestedSet<Artifact> directJars,
      NestedSet<Artifact> compileTimeDependencyArtifacts,
      Label targetLabel,
      @Nullable String injectingRuleKind,
      NestedSet<Artifact> excludedArtifacts,
      StrictDepsMode strictJavaDeps) {
    this.sourceFiles = sourceFiles;
    this.runtimeClassPath = runtimeClassPath;
    this.directJars = directJars;
    this.compileTimeClassPath = compileTimeClassPath;
    this.bootClassPath = bootClassPath;
    this.sourcePath = sourcePath;
    this.nativeLibraries = nativeLibraries;
    this.plugins = plugins;
    this.resources = resources;
    this.resourceJars = resourceJars;
    this.messages = messages;
    this.sourceJars = sourceJars;
    this.classPathResources = classPathResources;
    this.additionalOutputs = additionalOutputs;
    this.compileTimeDependencyArtifacts = compileTimeDependencyArtifacts;
    this.targetLabel = targetLabel;
    this.injectingRuleKind = injectingRuleKind;
    this.excludedArtifacts = excludedArtifacts;
    this.strictJavaDeps = strictJavaDeps;
  }

  JavaTargetAttributes withAdditionalClassPathEntries(
      NestedSet<Artifact> additionalClassPathEntries) {
    NestedSet<Artifact> compileTimeClassPath =
        NestedSetBuilder.fromNestedSet(additionalClassPathEntries)
            .addTransitive(this.compileTimeClassPath)
            .build();
    NestedSet<Artifact> directJars =
        NestedSetBuilder.fromNestedSet(additionalClassPathEntries)
            .addTransitive(this.directJars)
            .build();
    return new JavaTargetAttributes(
        sourceFiles,
        runtimeClassPath,
        compileTimeClassPath,
        bootClassPath,
        sourcePath,
        nativeLibraries,
        plugins,
        resources,
        resourceJars,
        messages,
        sourceJars,
        classPathResources,
        additionalOutputs,
        directJars,
        compileTimeDependencyArtifacts,
        targetLabel,
        injectingRuleKind,
        excludedArtifacts,
        strictJavaDeps);
  }

  public NestedSet<Artifact> getDirectJars() {
    return directJars;
  }

  public NestedSet<Artifact> getCompileTimeDependencyArtifacts() {
    return compileTimeDependencyArtifacts;
  }

  public ImmutableList<Artifact> getSourceJars() {
    return sourceJars;
  }

  public Map<PathFragment, Artifact> getResources() {
    return resources;
  }

  public NestedSet<Artifact> getResourceJars() {
    return resourceJars;
  }

  public List<Artifact> getMessages() {
    return messages;
  }

  public ImmutableList<Artifact> getClassPathResources() {
    return classPathResources;
  }

  public ImmutableSet<Artifact> getAdditionalOutputs() {
    return additionalOutputs;
  }

  private NestedSet<Artifact> getExcludedArtifacts() {
    return excludedArtifacts;
  }

  /**
   * Returns the artifacts needed on the runtime classpath of this target.
   *
   * <p>See also {@link #getRuntimeClassPathForArchive()}.
   */
  public NestedSet<Artifact> getRuntimeClassPath() {
    return runtimeClassPath;
  }

  /**
   * Returns the classpath artifacts needed in a deploy jar for this target.
   *
   * <p>This excludes the artifacts made available by jars in the deployment environment.
   */
  public NestedSet<Artifact> getRuntimeClassPathForArchive() {
    NestedSet<Artifact> runtimeClasspath = getRuntimeClassPath();

    if (getExcludedArtifacts().isEmpty()) {
      return runtimeClasspath;
    } else {
      return NestedSetBuilder.wrap(
          Order.STABLE_ORDER,
          Iterables.filter(
              runtimeClasspath.toList(),
              Predicates.not(Predicates.in(getExcludedArtifacts().toSet()))));
    }
  }

  public NestedSet<Artifact> getCompileTimeClassPath() {
    return compileTimeClassPath;
  }

  public NestedSet<Artifact> getBootClassPath() {
    return bootClassPath;
  }

  public ImmutableList<Artifact> getSourcePath() {
    return sourcePath;
  }

  public JavaPluginInfoProvider plugins() {
    return plugins;
  }

  public ImmutableSet<Artifact> getSourceFiles() {
    return sourceFiles;
  }

  public List<Artifact> getNativeLibraries() {
    return nativeLibraries;
  }

  public boolean hasSources() {
    return !sourceFiles.isEmpty() || !sourceJars.isEmpty();
  }

  public boolean hasResources() {
    return !resources.isEmpty()
        || !messages.isEmpty()
        || !classPathResources.isEmpty()
        || !resourceJars.isEmpty();
  }

  public boolean hasMessages() {
    return !messages.isEmpty();
  }

  public Label getTargetLabel() {
    return targetLabel;
  }

  @Nullable
  public String getInjectingRuleKind() {
    return injectingRuleKind;
  }

  public StrictDepsMode getStrictJavaDeps() {
    return strictJavaDeps;
  }
}
