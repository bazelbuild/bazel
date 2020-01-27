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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

/** Represents common aspects of all JVM targeting configured targets. */
public final class ClasspathConfiguredFragment {

  private final NestedSet<Artifact> runtimeClasspath;
  private final NestedSet<Artifact> compileTimeClasspath;
  private final NestedSet<Artifact> bootClasspath;

  /**
   * Initializes the runtime and compile time classpaths for this target. This method should be
   * called during {@code initializationHook()} once a {@link JavaTargetAttributes} object for this
   * target is fully initialized.
   *
   * @param attributes the processed attributes of this Java target
   * @param isNeverLink whether to leave runtimeClasspath empty
   */
  public ClasspathConfiguredFragment(
      JavaCompilationArtifacts javaArtifacts,
      JavaTargetAttributes attributes,
      boolean isNeverLink,
      NestedSet<Artifact> bootClasspath) {
    if (!isNeverLink) {
      runtimeClasspath = getRuntimeClasspathList(attributes, javaArtifacts);
    } else {
      runtimeClasspath = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    }
    compileTimeClasspath = attributes.getCompileTimeClassPath();
    this.bootClasspath = bootClasspath;
  }

  public ClasspathConfiguredFragment() {
    runtimeClasspath = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    compileTimeClasspath = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    bootClasspath = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  }

  /**
   * Returns the runtime class path. It consists of the concatenation of the instrumentation class
   * path, output jars and the runtime time class path of the transitive dependencies of this rule.
   *
   * @param attributes the processed attributes of this Java target
   * @return a {@List} of artifacts that comprise the runtime class path.
   */
  private NestedSet<Artifact> getRuntimeClasspathList(
      JavaTargetAttributes attributes, JavaCompilationArtifacts javaArtifacts) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.naiveLinkOrder();
    builder.addAll(javaArtifacts.getRuntimeJars());
    builder.addTransitive(attributes.getRuntimeClassPath());
    return builder.build();
  }

  /**
   * Returns the classpath to be passed to the JVM when running a target containing this fragment.
   */
  public NestedSet<Artifact> getRuntimeClasspath() {
    return runtimeClasspath;
  }

  /**
   * Returns the classpath to be passed to the Java compiler when compiling a target containing this
   * fragment.
   */
  public NestedSet<Artifact> getCompileTimeClasspath() {
    return compileTimeClasspath;
  }

  /**
   * Returns the classpath to be passed as a boot classpath to the Java compiler when compiling a
   * target containing this fragment.
   */
  public NestedSet<Artifact> getBootClasspath() {
    return bootClasspath;
  }
}
