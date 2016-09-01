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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/** The collection of source jars from the transitive closure. */
@AutoValue
@Immutable
public abstract class JavaSourceJarsProvider implements TransitiveInfoProvider {

  public static JavaSourceJarsProvider create(
      NestedSet<Artifact> transitiveSourceJars, Iterable<Artifact> sourceJars) {
    return new AutoValue_JavaSourceJarsProvider(
        transitiveSourceJars, ImmutableList.copyOf(sourceJars));
  }

  JavaSourceJarsProvider() {}

  /**
   * Returns all the source jars in the transitive closure, that can be reached by a chain of
   * JavaSourceJarsProvider instances.
   */
  public abstract NestedSet<Artifact> getTransitiveSourceJars();

  /** Return the source jars that are to be built when the target is on the command line. */
  public abstract ImmutableList<Artifact> getSourceJars();

  public static JavaSourceJarsProvider merge(Iterable<JavaSourceJarsProvider> providers) {
    NestedSetBuilder<Artifact> transitiveSourceJars = NestedSetBuilder.stableOrder();
    ImmutableList.Builder<Artifact> sourceJars = ImmutableList.builder();

    for (JavaSourceJarsProvider provider : providers) {
      transitiveSourceJars.addTransitive(provider.getTransitiveSourceJars());
      sourceJars.addAll(provider.getSourceJars());
    }
    return JavaSourceJarsProvider.create(transitiveSourceJars.build(), sourceJars.build());
  }
}
