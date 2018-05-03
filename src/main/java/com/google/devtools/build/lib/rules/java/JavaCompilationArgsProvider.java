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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.FileType;
import java.util.Collection;

/** An interface for objects that provide information on how to include them in Java builds. */
@AutoValue
@Immutable
@AutoCodec
public abstract class JavaCompilationArgsProvider implements TransitiveInfoProvider {

  @AutoCodec.Instantiator
  public static JavaCompilationArgsProvider create(
      JavaCompilationArgs javaCompilationArgs,
      JavaCompilationArgs recursiveJavaCompilationArgs,
      NestedSet<Artifact> compileTimeJavaDependencyArtifacts) {
    return new AutoValue_JavaCompilationArgsProvider(
        javaCompilationArgs, recursiveJavaCompilationArgs, compileTimeJavaDependencyArtifacts);
  }

  public static JavaCompilationArgsProvider create(
      JavaCompilationArgs javaCompilationArgs,
      JavaCompilationArgs recursiveJavaCompilationArgs) {
    return create(
        javaCompilationArgs,
        recursiveJavaCompilationArgs,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));
  }

  /**
   * Non-recursively collected Java compilation information, used when Strict Java Deps is enabled
   * to implement {@link #getDirectCompileTimeJars}.
   *
   * @deprecated use {@link #getDirectCompileTimeJars} instead.
   */
  @Deprecated
  public abstract JavaCompilationArgs getJavaCompilationArgs();

  /**
   * Returns recursively collected Java compilation information.
   *
   * @deprecated use one of: {@link #getTransitiveCompileTimeJars}, {@link #getRuntimeJars}, {@link
   *     #getInstrumentationMetadata} instead.
   */
  @Deprecated
  public abstract JavaCompilationArgs getRecursiveJavaCompilationArgs();

  /**
   * Returns non-recursively collected compile-time jars. This is the set of jars that compilations
   * are permitted to reference with Strict Java Deps enabled.
   */
  public NestedSet<Artifact> getDirectCompileTimeJars() {
    return getJavaCompilationArgs().getCompileTimeJars();
  }

  /**
   * Returns non-recursively collected, non-interface compile-time jars.
   *
   * <p>If you're reading this, you probably want {@link #getTransitiveCompileTimeJars}.
   */
  public NestedSet<Artifact> getFullCompileTimeJars() {
    return getJavaCompilationArgs().getFullCompileTimeJars();
  }

  /**
   * Returns recursively collected compile-time jars. This is the compile-time classpath passed to
   * the compiler.
   */
  public NestedSet<Artifact> getTransitiveCompileTimeJars() {
    return getRecursiveJavaCompilationArgs().getCompileTimeJars();
  }

  /** Returns recursively collected, non-interface compile-time jars. */
  public NestedSet<Artifact> getFullTransitiveCompileTimeJars() {
    return getRecursiveJavaCompilationArgs().getFullCompileTimeJars();
  }

  /** Returns recursively collected runtime jars. */
  public NestedSet<Artifact> getRuntimeJars() {
    return getRecursiveJavaCompilationArgs().getRuntimeJars();
  }

  /** Returns recursively collected instrumentation metadata. */
  public NestedSet<Artifact> getInstrumentationMetadata() {
    return getRecursiveJavaCompilationArgs().getInstrumentationMetadata();
  }

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

  public static JavaCompilationArgsProvider merge(
      Collection<JavaCompilationArgsProvider> providers) {
    if (providers.size() == 1) {
      return Iterables.get(providers, 0);
    }

    JavaCompilationArgs.Builder javaCompilationArgs = JavaCompilationArgs.builder();
    JavaCompilationArgs.Builder recursiveJavaCompilationArgs = JavaCompilationArgs.builder();
    NestedSetBuilder<Artifact> compileTimeJavaDepArtifacts = NestedSetBuilder.stableOrder();

    for (JavaCompilationArgsProvider provider : providers) {
      javaCompilationArgs.addTransitiveArgs(
          provider.getJavaCompilationArgs(), JavaCompilationArgs.ClasspathType.BOTH);
      recursiveJavaCompilationArgs.addTransitiveArgs(
          provider.getRecursiveJavaCompilationArgs(), JavaCompilationArgs.ClasspathType.BOTH);
      compileTimeJavaDepArtifacts.addTransitive(provider.getCompileTimeJavaDependencyArtifacts());
    }

    return JavaCompilationArgsProvider.create(
        javaCompilationArgs.build(),
        recursiveJavaCompilationArgs.build(),
        compileTimeJavaDepArtifacts.build());
  }

  /**
   * Returns a {@link JavaCompilationArgsProvider} for the given {@link TransitiveInfoCollection}s.
   *
   * <p>If the given targets have a {@link JavaCompilationArgsProvider}, the information from that
   * provider will be returned. Otherwise, any jar files provided by the targets will be wrapped in
   * the returned provider.
   *
   * @deprecated The handling of raw jar files is present for legacy compatibility. All new
   *     Java-based rules should require their dependencies to provide {@link
   *     JavaCompilationArgsProvider}, and that precompiled jar files be wrapped in {@code
   *     java_import}. New rules should not use this method, and existing rules should be cleaned up
   *     to disallow jar files in their deps.
   */
  // TODO(b/11285003): disallow jar files in deps, require java_import instead
  @Deprecated
  public static JavaCompilationArgsProvider legacyFromTargets(
      Iterable<? extends TransitiveInfoCollection> infos) {
    JavaCompilationArgs.Builder argsBuilder = JavaCompilationArgs.builder();
    JavaCompilationArgs.Builder recursiveArgsBuilder = JavaCompilationArgs.builder();
    for (TransitiveInfoCollection info : infos) {
      JavaCompilationArgsProvider provider =
          JavaInfo.getProvider(JavaCompilationArgsProvider.class, info);
      if (provider != null) {
        argsBuilder.addTransitiveArgs(provider.getJavaCompilationArgs(), ClasspathType.BOTH);
        recursiveArgsBuilder.addTransitiveArgs(
            provider.getRecursiveJavaCompilationArgs(), ClasspathType.BOTH);
      } else {
        NestedSet<Artifact> filesToBuild = info.getProvider(FileProvider.class).getFilesToBuild();
        for (Artifact jar : FileType.filter(filesToBuild, JavaSemantics.JAR)) {
          argsBuilder.addRuntimeJar(jar).addCompileTimeJarAsFullJar(jar);
          recursiveArgsBuilder.addRuntimeJar(jar).addCompileTimeJarAsFullJar(jar);
        }
      }
    }
    return create(argsBuilder.build(), recursiveArgsBuilder.build());
  }
}
