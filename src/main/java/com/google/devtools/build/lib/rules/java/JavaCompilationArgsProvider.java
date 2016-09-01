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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * An interface for objects that provide information on how to include them in
 * Java builds.
 */
@AutoValue
@Immutable
public abstract class JavaCompilationArgsProvider implements TransitiveInfoProvider {

  public static JavaCompilationArgsProvider create(
      JavaCompilationArgs javaCompilationArgs,
      JavaCompilationArgs recursiveJavaCompilationArgs,
      NestedSet<Artifact> compileTimeJavaDepArtifacts,
      NestedSet<Artifact> runTimeJavaDepArtifacts) {
    return new AutoValue_JavaCompilationArgsProvider(
        javaCompilationArgs,
        recursiveJavaCompilationArgs,
        compileTimeJavaDepArtifacts,
        runTimeJavaDepArtifacts);
  }

  public static JavaCompilationArgsProvider create(
      JavaCompilationArgs javaCompilationArgs,
      JavaCompilationArgs recursiveJavaCompilationArgs) {
    return create(
        javaCompilationArgs,
        recursiveJavaCompilationArgs,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));
  }

  /**
   * Returns non-recursively collected Java compilation information for
   * building this target (called when strict_java_deps = 1).
   *
   * <p>Note that some of the parameters are still collected from the complete
   * transitive closure. The non-recursive collection applies mainly to
   * compile-time jars.
   */
  public abstract JavaCompilationArgs getJavaCompilationArgs();

  /**
   * Returns recursively collected Java compilation information for building
   * this target (called when strict_java_deps = 0).
   */
  public abstract JavaCompilationArgs getRecursiveJavaCompilationArgs();

  /**
   * Returns non-recursively collected Java dependency artifacts for
   * computing a restricted classpath when building this target (called when
   * strict_java_deps = 1).
   *
   * <p>Note that dependency artifacts are needed only when non-recursive
   * compilation args do not provide a safe super-set of dependencies.
   * Non-strict targets such as proto_library, always collecting their
   * transitive closure of deps, do not need to provide dependency artifacts.
   */
  public abstract NestedSet<Artifact> getCompileTimeJavaDependencyArtifacts();

  /**
   * Returns Java dependency artifacts for computing a restricted run-time
   * classpath (called when strict_java_deps = 1).
   */
  public abstract NestedSet<Artifact> getRunTimeJavaDependencyArtifacts();

  public static JavaCompilationArgsProvider merge(Iterable<JavaCompilationArgsProvider> providers) {
    JavaCompilationArgs.Builder javaCompilationArgs = JavaCompilationArgs.builder();
    JavaCompilationArgs.Builder recursiveJavaCompilationArgs = JavaCompilationArgs.builder();
    NestedSetBuilder<Artifact> compileTimeJavaDepArtifacts = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> runTimeJavaDepArtifacts = NestedSetBuilder.stableOrder();

    for (JavaCompilationArgsProvider provider : providers) {
      javaCompilationArgs.addTransitiveArgs(
          provider.getJavaCompilationArgs(), JavaCompilationArgs.ClasspathType.BOTH);
      recursiveJavaCompilationArgs.addTransitiveArgs(
          provider.getRecursiveJavaCompilationArgs(), JavaCompilationArgs.ClasspathType.BOTH);
      compileTimeJavaDepArtifacts.addTransitive(provider.getCompileTimeJavaDependencyArtifacts());
      runTimeJavaDepArtifacts.addTransitive(provider.getRunTimeJavaDependencyArtifacts());
    }

    return JavaCompilationArgsProvider.create(
        javaCompilationArgs.build(),
        recursiveJavaCompilationArgs.build(),
        compileTimeJavaDepArtifacts.build(),
        runTimeJavaDepArtifacts.build());
  }
}
