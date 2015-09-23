// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import javax.annotation.Nullable;

/**
 * The collection of gen jars from the transitive closure.
 */
@Immutable
public final class JavaGenJarsProvider implements TransitiveInfoProvider {

  private final boolean usesAnnotationProcessing;
  @Nullable
  private final Artifact genClassJar;
  @Nullable
  private final Artifact genSourceJar;
  private final NestedSet<Artifact> transitiveGenClassJars;
  private final NestedSet<Artifact> transitiveGenSourceJars;

  public JavaGenJarsProvider(
      boolean usesAnnotationProcessing,
      @Nullable Artifact genClassJar,
      @Nullable Artifact genSourceJar,
      NestedSet<Artifact> transitiveGenClassJars,
      NestedSet<Artifact> transitiveGenSourceJars) {
    this.usesAnnotationProcessing = usesAnnotationProcessing;
    this.genClassJar = genClassJar;
    this.genSourceJar = genSourceJar;
    this.transitiveGenClassJars = transitiveGenClassJars;
    this.transitiveGenSourceJars = transitiveGenSourceJars;
  }

  public boolean usesAnnotationProcessing() {
    return usesAnnotationProcessing;
  }

  @Nullable
  public Artifact getGenClassJar() {
    return genClassJar;
  }

  @Nullable
  public Artifact getGenSourceJar() {
    return genSourceJar;
  }

  public NestedSet<Artifact> getTransitiveGenClassJars() {
    return transitiveGenClassJars;
  }

  public NestedSet<Artifact> getTransitiveGenSourceJars() {
    return transitiveGenSourceJars;
  }
}
