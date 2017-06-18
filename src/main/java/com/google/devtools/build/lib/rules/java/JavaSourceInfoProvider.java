// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Map;

/**
 * A Provider describing the java sources directly belonging to a java rule.
 */
@Immutable
public final class JavaSourceInfoProvider implements TransitiveInfoProvider {
  private final Collection<Artifact> sourceFiles;
  private final Collection<Artifact> sourceJars;
  private final Collection<Artifact> jarFiles;
  private final Collection<Artifact> sourceJarsForJarFiles;
  private final Map<PathFragment, Artifact> resources;
  private final Collection<String> processorNames;
  private final NestedSet<Artifact> processorPath;

  private JavaSourceInfoProvider(
      Collection<Artifact> sourceFiles,
      Collection<Artifact> sourceJars,
      Collection<Artifact> jarFiles,
      Collection<Artifact> sourceJarsForJarFiles,
      Map<PathFragment, Artifact> resources,
      Collection<String> processorNames,
      NestedSet<Artifact> processorPath) {
    this.sourceFiles = sourceFiles;
    this.sourceJars = sourceJars;
    this.jarFiles = jarFiles;
    this.sourceJarsForJarFiles = sourceJarsForJarFiles;
    this.resources = resources;
    this.processorNames = processorNames;
    this.processorPath = processorPath;
  }

  /** Gets the original Java source files provided as inputs to this rule. */
  public Collection<Artifact> getSourceFiles() {
    return sourceFiles;
  }

  /**
   * Gets the original source jars provided as inputs to this rule.
   *
   * <p>These should contain Java source files, but can contain other files as well.
   */
  public Collection<Artifact> getSourceJars() {
    return sourceJars;
  }

  /**
   * Gets the original pre-built jars provided as inputs to this rule.
   *
   * <p>These should be used where .class files are needed or wanted in place of recompiling the
   * sources from {@link #getSourceJarsForJarFiles()}, as this is the source of truth used by the
   * normal Java machinery.
   */
  public Collection<Artifact> getJarFiles() {
    return jarFiles;
  }

  /**
   * Gets the source jars containing the sources of the jars contained in {@link #getJarFiles}.
   *
   * <p>These should be used in place of {@link #getJarFiles()} if and only if source is required.
   */
  public Collection<Artifact> getSourceJarsForJarFiles() {
    return sourceJarsForJarFiles;
  }

  /**
   * Gets the Java resources which were included in this rule's output.
   *
   * <p>Each key in the map (a path within the jar) should correspond to the artifact which belongs
   * at that path. The path fragment should be some suffix of the artifact's exec path.
   */
  public Map<PathFragment, Artifact> getResources() {
    return resources;
  }

  /** Gets the names of the annotation processors which operate on this rule's sources. */
  public Collection<String> getProcessorNames() {
    return processorNames;
  }

  /** Gets the classpath for the annotation processors which operate on this rule's sources. */
  public NestedSet<Artifact> getProcessorPath() {
    return processorPath;
  }

  /**
   * Constructs a JavaSourceInfoProvider using the sources in the given JavaTargetAttributes.
   *
   * @param attributes the object from which to draw the sources
   * @param semantics semantics used to find the path for a resource within the jar
   */
  public static JavaSourceInfoProvider fromJavaTargetAttributes(
      JavaTargetAttributes attributes, JavaSemantics semantics) {
    return new Builder()
        .setSourceFiles(attributes.getSourceFiles())
        .setSourceJars(attributes.getSourceJars())
        .setResources(attributes.getResources())
        .setProcessorNames(attributes.getProcessorNames())
        .setProcessorPath(attributes.getProcessorPath())
        .build();
  }

  /** Builder class for constructing JavaSourceInfoProviders. */
  public static final class Builder {
    private Collection<Artifact> sourceFiles = ImmutableList.<Artifact>of();
    private Collection<Artifact> sourceJars = ImmutableList.<Artifact>of();
    private Collection<Artifact> jarFiles = ImmutableList.<Artifact>of();
    private Collection<Artifact> sourceJarsForJarFiles = ImmutableList.<Artifact>of();
    private Map<PathFragment, Artifact> resources = ImmutableMap.<PathFragment, Artifact>of();
    private Collection<String> processorNames = ImmutableList.<String>of();
    private NestedSet<Artifact> processorPath = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);

    /** Sets the source files included as part of the sources of this rule. */
    public Builder setSourceFiles(Collection<Artifact> sourceFiles) {
      this.sourceFiles = Preconditions.checkNotNull(sourceFiles);
      return this;
    }

    /** Sets the source jars included as part of the sources of this rule. */
    public Builder setSourceJars(Collection<Artifact> sourceJars) {
      this.sourceJars = Preconditions.checkNotNull(sourceJars);
      return this;
    }

    /**
     * Sets the pre-built jar files included as part of the sources of this rule.
     */
    public Builder setJarFiles(Collection<Artifact> jarFiles) {
      this.jarFiles = Preconditions.checkNotNull(jarFiles);
      return this;
    }

    /**
     * Sets the source jars corresponding to the jar files included in this rule.
     *
     * <p>Used by, e.g., the srcjars attribute of {@link JavaImport}.
     */
    public Builder setSourceJarsForJarFiles(Collection<Artifact> sourceJarsForJarFiles) {
      this.sourceJarsForJarFiles = Preconditions.checkNotNull(sourceJarsForJarFiles);
      return this;
    }

    /**
     * Sets the resources included in this rule.
     *
     * <p>Each key in the map (a path within the jar) should correspond to the artifact which
     * belongs at that path. The path fragment should be some tail of the artifact's exec path.
     */
    public Builder setResources(Map<PathFragment, Artifact> resources) {
      this.resources = Preconditions.checkNotNull(resources);
      return this;
    }

    /** Sets the names of the annotation processors used by this rule. */
    public Builder setProcessorNames(Collection<String> processorNames) {
      this.processorNames = Preconditions.checkNotNull(processorNames);
      return this;
    }

    /** Sets the classpath used by this rule for annotation processing. */
    public Builder setProcessorPath(NestedSet<Artifact> processorPath) {
      Preconditions.checkNotNull(processorPath);
      this.processorPath = processorPath;
      return this;
    }

    /** Constructs the JavaSourceInfoProvider from the provided Java sources. */
    public JavaSourceInfoProvider build() {
      return new JavaSourceInfoProvider(
          sourceFiles,
          sourceJars,
          jarFiles,
          sourceJarsForJarFiles,
          resources,
          processorNames,
          processorPath);
    }
  }
}
