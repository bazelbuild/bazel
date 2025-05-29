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

import static java.util.Objects.requireNonNull;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.java.JavaInfo.JavaInfoInternalProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * A collection of recursively collected Java build information.
 *
 * @param runtimeJars Returns recursively collected runtime jars.
 * @param directCompileTimeJars Returns non-recursively collected compile-time jars. This is the set
 *     of jars that compilations are permitted to reference with Strict Java Deps enabled.
 *     <p>If you're reading this, you probably want {@link #getTransitiveCompileTimeJars} .
 * @param transitiveCompileTimeJars Returns recursively collected compile-time jars. This is the
 *     compile-time classpath passed to the compiler.
 * @param directFullCompileTimeJars Returns non-recursively collected, non-interface compile-time
 *     jars.
 *     <p>If you're reading this, you probably want {@link #getTransitiveCompileTimeJars} .
 * @param transitiveFullCompileTimeJars Returns recursively collected, non-interface compile-time
 *     jars.
 *     <p>If you're reading this, you probably want {@link #getTransitiveCompileTimeJars} .
 * @param compileTimeJavaDependencyArtifacts Returns non-recursively collected Java dependency
 *     artifacts for computing a restricted classpath when building this target (called when
 *     strict_java_deps = 1).
 *     <p>Note that dependency artifacts are needed only when non-recursive compilation args do not
 *     provide a safe super-set of dependencies. Non-strict targets such as proto_library, always
 *     collecting their transitive closure of deps, do not need to provide dependency artifacts.
 */
@Immutable
@AutoCodec
public record JavaCompilationArgsProvider(
    NestedSet<Artifact> runtimeJars,
    NestedSet<Artifact> directCompileTimeJars,
    NestedSet<Artifact> transitiveCompileTimeJars,
    NestedSet<Artifact> directFullCompileTimeJars,
    NestedSet<Artifact> transitiveFullCompileTimeJars,
    NestedSet<Artifact> compileTimeJavaDependencyArtifacts,
    NestedSet<Artifact> directHeaderCompilationJars)
    implements JavaInfoInternalProvider {
  public JavaCompilationArgsProvider {
    requireNonNull(runtimeJars, "runtimeJars");
    requireNonNull(directCompileTimeJars, "directCompileTimeJars");
    requireNonNull(transitiveCompileTimeJars, "transitiveCompileTimeJars");
    requireNonNull(directFullCompileTimeJars, "directFullCompileTimeJars");
    requireNonNull(transitiveFullCompileTimeJars, "transitiveFullCompileTimeJars");
    requireNonNull(compileTimeJavaDependencyArtifacts, "compileTimeJavaDependencyArtifacts");
    requireNonNull(directHeaderCompilationJars, "directHeaderCompilationJars");
  }

  @SerializationConstant
  public static final JavaCompilationArgsProvider EMPTY =
      create(
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER));

  private static JavaCompilationArgsProvider create(
      NestedSet<Artifact> runtimeJars,
      NestedSet<Artifact> directCompileTimeJars,
      NestedSet<Artifact> transitiveCompileTimeJars,
      NestedSet<Artifact> directFullCompileTimeJars,
      NestedSet<Artifact> transitiveFullCompileTimeJars,
      NestedSet<Artifact> compileTimeJavaDependencyArtifacts,
      NestedSet<Artifact> directHeaderCompilationJars) {
    return new JavaCompilationArgsProvider(
        runtimeJars,
        directCompileTimeJars,
        transitiveCompileTimeJars,
        directFullCompileTimeJars,
        transitiveFullCompileTimeJars,
        compileTimeJavaDependencyArtifacts,
        directHeaderCompilationJars);
  }

  /**
   * Constructs a {@link JavaCompilationArgsProvider} instance for a Starlark-constructed {@link
   * JavaInfo}.
   *
   * @param javaInfo the {@link JavaInfo} instance from which to extract the relevant fields
   * @return a {@link JavaCompilationArgsProvider} instance, or {@code null} if this is a {@link
   *     JavaInfo} for a {@code java_binary} or {@code java_test}
   * @throws EvalException if there were errors reading any fields
   * @throws TypeException if some field was not a {@link Depset} of {@link Artifact}s
   */
  @Nullable
  static JavaCompilationArgsProvider fromStarlarkJavaInfo(StructImpl javaInfo)
      throws EvalException, TypeException {
    Boolean isBinary = javaInfo.getValue("_is_binary", Boolean.class);
    if (isBinary != null && isBinary) {
      return null;
    }
    return create(
        /* runtimeJars= */ getDepset(javaInfo, "transitive_runtime_jars"),
        /* directCompileTimeJars= */ getDepset(javaInfo, "compile_jars"),
        /* transitiveCompileTimeJars= */ getDepset(javaInfo, "transitive_compile_time_jars"),
        /* directFullCompileTimeJars= */ getDepset(javaInfo, "full_compile_jars"),
        /* transitiveFullCompileTimeJars= */ getDepset(
            javaInfo, "_transitive_full_compile_time_jars"),
        /* compileTimeJavaDependencyArtifacts= */ getDepset(
            javaInfo, "_compile_time_java_dependencies"),
        /* directHeaderCompilationJars= */ maybeGetDepset(
            javaInfo, "header_compilation_direct_deps"));
  }

  // TODO: b/417791104 - make this unconditional once Bazel 8.3.0 is released
  private static final NestedSet<Artifact> maybeGetDepset(StructImpl javaInfo, String name)
      throws EvalException, TypeException {
    Depset depset = javaInfo.getValue(name, Depset.class);
    if (depset == null) {
      return NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    }
    return depset.getSet(Artifact.class);
  }

  private static final NestedSet<Artifact> getDepset(StructImpl javaInfo, String name)
      throws EvalException, TypeException {
    return javaInfo.getValue(name, Depset.class).getSet(Artifact.class);
  }
}
