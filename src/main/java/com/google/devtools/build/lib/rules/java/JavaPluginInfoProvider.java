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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provider for users of Java plugins.
 */
@Immutable
public final class JavaPluginInfoProvider implements TransitiveInfoProvider {

  public static JavaPluginInfoProvider merge(Iterable<JavaPluginInfoProvider> providers) {
    ImmutableSet.Builder<String> processorClasses = ImmutableSet.builder();
    NestedSetBuilder<Artifact> processorClasspath = NestedSetBuilder.naiveLinkOrder();
    ImmutableSet.Builder<String> apiGeneratingProcessorClasses = ImmutableSet.builder();
    NestedSetBuilder<Artifact> apiGeneratingProcessorClasspath = NestedSetBuilder.naiveLinkOrder();

    for (JavaPluginInfoProvider provider : providers) {
      processorClasses.addAll(provider.getProcessorClasses());
      processorClasspath.addTransitive(provider.getProcessorClasspath());
      apiGeneratingProcessorClasses.addAll(provider.getApiGeneratingProcessorClasses());
      apiGeneratingProcessorClasspath.addTransitive(provider.getApiGeneratingProcessorClasspath());
    }
    return new JavaPluginInfoProvider(
        processorClasses.build(),
        processorClasspath.build(),
        apiGeneratingProcessorClasses.build(),
        apiGeneratingProcessorClasspath.build());
  }

  private final ImmutableSet<String> processorClasses;
  private final NestedSet<Artifact> processorClasspath;
  private final ImmutableSet<String> apiGeneratingProcessorClasses;
  private final NestedSet<Artifact> apiGeneratingProcessorClasspath;

  public JavaPluginInfoProvider(
      ImmutableSet<String> processorClasses,
      NestedSet<Artifact> processorClasspath,
      ImmutableSet<String> apiGeneratingProcessorClasses,
      NestedSet<Artifact> apiGeneratingProcessorClasspath) {
    this.processorClasses = processorClasses;
    this.processorClasspath = processorClasspath;
    this.apiGeneratingProcessorClasses = apiGeneratingProcessorClasses;
    this.apiGeneratingProcessorClasspath = apiGeneratingProcessorClasspath;
  }

  /**
   * Returns the class that should be passed to javac in order
   * to run the annotation processor this class represents.
   */
  public ImmutableSet<String> getProcessorClasses() {
    return processorClasses;
  }

  /**
   * Returns the artifacts to add to the runtime classpath for this plugin.
   */
  public NestedSet<Artifact> getProcessorClasspath() {
    return processorClasspath;
  }

  /** Returns the class names of API-generating annotation processors. */
  public ImmutableSet<String> getApiGeneratingProcessorClasses() {
    return apiGeneratingProcessorClasses;
  }

  /** Returns the artifacts to add to the runtime classpath of the API-generating processors. */
  public NestedSet<Artifact> getApiGeneratingProcessorClasspath() {
    return apiGeneratingProcessorClasspath;
  }
}
