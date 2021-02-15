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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.Collection;

/** A Provider describing the java sources directly belonging to a java rule. */
@Immutable
@AutoCodec
public final class JavaSourceInfoProvider implements TransitiveInfoProvider {
  private final Collection<Artifact> sourceFiles;
  private final Collection<Artifact> sourceJars;

  @VisibleForSerialization
  JavaSourceInfoProvider(Collection<Artifact> sourceFiles, Collection<Artifact> sourceJars) {
    this.sourceFiles = sourceFiles;
    this.sourceJars = sourceJars;
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
        .build();
  }

  public static JavaSourceInfoProvider merge(Collection<JavaSourceInfoProvider> sourceInfos) {
    JavaSourceInfoProvider.Builder javaSourceInfo = new JavaSourceInfoProvider.Builder();
    ImmutableList.Builder<Artifact> sourceFiles = new ImmutableList.Builder<>();
    ImmutableList.Builder<Artifact> sourceJars = new ImmutableList.Builder<>();

    for (JavaSourceInfoProvider sourceInfo : sourceInfos) {
      sourceFiles.addAll(sourceInfo.getSourceFiles());
      sourceJars.addAll(sourceInfo.getSourceJars());
    }
    javaSourceInfo.setSourceFiles(sourceFiles.build());
    javaSourceInfo.setSourceJars(sourceJars.build());
    return javaSourceInfo.build();
  }

  /** Builder class for constructing JavaSourceInfoProviders. */
  public static final class Builder {
    private Collection<Artifact> sourceFiles = ImmutableList.<Artifact>of();
    private Collection<Artifact> sourceJars = ImmutableList.<Artifact>of();

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

    /** Constructs the JavaSourceInfoProvider from the provided Java sources. */
    public JavaSourceInfoProvider build() {
      return new JavaSourceInfoProvider(sourceFiles, sourceJars);
    }
  }
}
