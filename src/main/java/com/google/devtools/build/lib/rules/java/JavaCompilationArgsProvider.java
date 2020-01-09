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
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.FileType;
import java.util.Collection;
import java.util.Iterator;

/** A collection of recursively collected Java build information. */
@AutoValue
@Immutable
@AutoCodec
public abstract class JavaCompilationArgsProvider implements TransitiveInfoProvider {

  @AutoCodec
  public static final JavaCompilationArgsProvider EMPTY =
      create(
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.create(Order.NAIVE_LINK_ORDER));

  @AutoCodec.Instantiator
  public static JavaCompilationArgsProvider create(
      NestedSet<Artifact> runtimeJars,
      NestedSet<Artifact> directCompileTimeJars,
      NestedSet<Artifact> transitiveCompileTimeJars,
      NestedSet<Artifact> directFullCompileTimeJars,
      NestedSet<Artifact> transitiveFullCompileTimeJars,
      NestedSet<Artifact> compileTimeJavaDependencyArtifacts) {
    return new AutoValue_JavaCompilationArgsProvider(
        runtimeJars,
        directCompileTimeJars,
        transitiveCompileTimeJars,
        directFullCompileTimeJars,
        transitiveFullCompileTimeJars,
        compileTimeJavaDependencyArtifacts);
  }

  /** Returns recursively collected runtime jars. */
  public abstract NestedSet<Artifact> getRuntimeJars();

  /**
   * Returns non-recursively collected compile-time jars. This is the set of jars that compilations
   * are permitted to reference with Strict Java Deps enabled.
   *
   * <p>If you're reading this, you probably want {@link #getTransitiveCompileTimeJars}.
   */
  public abstract NestedSet<Artifact> getDirectCompileTimeJars();

  /**
   * Returns recursively collected compile-time jars. This is the compile-time classpath passed to
   * the compiler.
   */
  public abstract NestedSet<Artifact> getTransitiveCompileTimeJars();

  /**
   * Returns non-recursively collected, non-interface compile-time jars.
   *
   * <p>If you're reading this, you probably want {@link #getTransitiveCompileTimeJars}.
   */
  public abstract NestedSet<Artifact> getDirectFullCompileTimeJars();

  /**
   * Returns recursively collected, non-interface compile-time jars.
   *
   * <p>If you're reading this, you probably want {@link #getTransitiveCompileTimeJars}.
   */
  public abstract NestedSet<Artifact> getTransitiveFullCompileTimeJars();

  /**
   * Returns non-recursively collected Java dependency artifacts for computing a restricted
   * classpath when building this target (called when strict_java_deps = 1).
   *
   * <p>Note that dependency artifacts are needed only when non-recursive compilation args do not
   * provide a safe super-set of dependencies. Non-strict targets such as proto_library, always
   * collecting their transitive closure of deps, do not need to provide dependency artifacts.
   */
  public abstract NestedSet<Artifact> getCompileTimeJavaDependencyArtifacts();

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
    return legacyFromTargets(infos, /* javaProtoLibraryStrictDeps= */ false);
  }

  @Deprecated
  public static JavaCompilationArgsProvider legacyFromTargets(
      Iterable<? extends TransitiveInfoCollection> infos, boolean javaProtoLibraryStrictDeps) {
    Builder argsBuilder = builder();
    for (TransitiveInfoCollection info : infos) {
      JavaCompilationArgsProvider provider = null;

      if (javaProtoLibraryStrictDeps) {
        JavaStrictCompilationArgsProvider strictCompilationArgsProvider =
            JavaInfo.getProvider(JavaStrictCompilationArgsProvider.class, info);
        if (strictCompilationArgsProvider != null) {
          provider = strictCompilationArgsProvider.getJavaCompilationArgsProvider();
        }
      }
      if (provider == null) {
        provider = JavaInfo.getProvider(JavaCompilationArgsProvider.class, info);
      }
      if (provider != null) {
        argsBuilder.addExports(provider);
      } else {
        NestedSet<Artifact> filesToBuild = info.getProvider(FileProvider.class).getFilesToBuild();
        for (Artifact jar : FileType.filter(filesToBuild.toList(), JavaSemantics.JAR)) {
          argsBuilder
              .addRuntimeJar(jar)
              .addDirectCompileTimeJar(/* interfaceJar= */ jar, /* fullJar= */ jar);
        }
      }
    }
    return argsBuilder.build();
  }

  /** Enum to specify transitive compilation args traversal */
  public enum ClasspathType {
    /* treat the same for compile time and runtime */
    BOTH,

    /* Only include on compile classpath */
    COMPILE_ONLY,

    /* Only include on runtime classpath */
    RUNTIME_ONLY
  }

  /**
   * Disable strict deps enforcement for the given {@link JavaCompilationArgsProvider}; the direct
   * jars in the result include the full transitive compile-time classpath from the input.
   */
  public static JavaCompilationArgsProvider makeNonStrict(JavaCompilationArgsProvider args) {
    // Omit jdeps, which aren't available transitively and aren't useful for reduced classpath
    // pruning for non-strict targets: the direct classpath and transitive classpath are the same,
    // so there's nothing to prune, and reading jdeps at compile-time isn't free.
    return builder()
        .addDirectCompileTimeJars(
            /* interfaceJars= */ args.getTransitiveCompileTimeJars(),
            /* fullJars= */ args.getTransitiveFullCompileTimeJars())
        .addRuntimeJars(args.getRuntimeJars())
        .build();
  }

  /**
   * Returns a {@link JavaCompilationArgsProvider} that forwards the union of information from the
   * inputs. Direct deps of the inputs are merged into the direct deps of the outputs.
   *
   * <p>This is morally equivalent to an exports-only {@code java_import} rule that forwards some
   * dependencies.
   */
  public static JavaCompilationArgsProvider merge(Iterable<JavaCompilationArgsProvider> providers) {
    Iterator<JavaCompilationArgsProvider> it = providers.iterator();
    if (!it.hasNext()) {
      return EMPTY;
    }
    JavaCompilationArgsProvider first = it.next();
    if (!it.hasNext()) {
      return first;
    }
    Builder javaCompilationArgs = builder();
    javaCompilationArgs.addExports(first);
    do {
      javaCompilationArgs.addExports(it.next());
    } while (it.hasNext());
    return javaCompilationArgs.build();
  }

  /**
   * Returns a {@link JavaCompilationArgsProvider} that forwards the union of information from the
   * inputs, see {@link #merge(Collection<JavaCompilationArgsProvider>)}.
   */
  public static JavaCompilationArgsProvider merge(JavaCompilationArgsProvider... providers) {
    return merge(ImmutableList.copyOf(providers));
  }

  /** Returns a new builder instance. */
  public static final Builder builder() {
    return new Builder();
  }

  /** A {@link JavaCompilationArgsProvider}Builder. */
  public static final class Builder {
    private final NestedSetBuilder<Artifact> runtimeJarsBuilder = NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> directCompileTimeJarsBuilder =
        NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> transitiveCompileTimeJarsBuilder =
        NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> directFullCompileTimeJarsBuilder =
        NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> transitiveFullCompileTimeJarsBuilder =
        NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> compileTimeJavaDependencyArtifactsBuilder =
        NestedSetBuilder.naiveLinkOrder();

    /** Use {@code TransitiveJavaCompilationArgs#builder()} to instantiate the builder. */
    private Builder() {}

    /**
     * Legacy method for dealing with objects which construct {@link JavaCompilationArtifacts}
     * objects.
     */
    // TODO(bazel-team): Remove when we get rid of JavaCompilationArtifacts.
    public Builder merge(JavaCompilationArtifacts other, boolean isNeverLink) {
      if (!isNeverLink) {
        addRuntimeJars(NestedSetBuilder.wrap(Order.NAIVE_LINK_ORDER, other.getRuntimeJars()));
      }
      addDirectCompileTimeJars(
          /* interfaceJars= */ NestedSetBuilder.wrap(
              Order.NAIVE_LINK_ORDER, other.getCompileTimeJars()),
          /* fullJars= */ NestedSetBuilder.wrap(
              Order.NAIVE_LINK_ORDER, other.getFullCompileTimeJars()));
      return this;
    }

    /**
     * Legacy method for dealing with objects which construct {@link JavaCompilationArtifacts}
     * objects.
     */
    public Builder merge(JavaCompilationArtifacts other) {
      return merge(other, /* isNeverLink= */ false);
    }

    public Builder addRuntimeJar(Artifact runtimeJar) {
      this.runtimeJarsBuilder.add(runtimeJar);
      return this;
    }

    public Builder addRuntimeJars(NestedSet<Artifact> runtimeJars) {
      this.runtimeJarsBuilder.addTransitive(runtimeJars);
      return this;
    }

    /** Adds a pair of direct interface and implementation jars. */
    public Builder addDirectCompileTimeJar(Artifact interfaceJar, Artifact fullJar) {
      this.directCompileTimeJarsBuilder.add(interfaceJar);
      this.transitiveCompileTimeJarsBuilder.add(interfaceJar);
      this.directFullCompileTimeJarsBuilder.add(fullJar);
      this.transitiveFullCompileTimeJarsBuilder.add(fullJar);
      return this;
    }

    /** Adds paired sets of direct interface and implementation jars. */
    public Builder addDirectCompileTimeJars(
        NestedSet<Artifact> interfaceJars, NestedSet<Artifact> fullJars) {
      this.directCompileTimeJarsBuilder.addTransitive(interfaceJars);
      this.transitiveCompileTimeJarsBuilder.addTransitive(interfaceJars);
      this.directFullCompileTimeJarsBuilder.addTransitive(fullJars);
      this.transitiveFullCompileTimeJarsBuilder.addTransitive(fullJars);
      return this;
    }

    /**
     * Adds transitive interface compile-time jars.
     *
     * @deprecated this is necessary to support java_common.create_provider, which is also
     *     deprecated. It allows creating providers where the direct compile-time jars aren't a
     *     subset of the transitive jars, and it doesn't provide a way to associate the 'full' jars.
     */
    @Deprecated
    public Builder addTransitiveCompileTimeJars(NestedSet<Artifact> transitiveCompileTimeJars) {
      this.transitiveCompileTimeJarsBuilder.addTransitive(transitiveCompileTimeJars);
      return this;
    }

    public Builder addCompileTimeJavaDependencyArtifacts(
        NestedSet<Artifact> compileTimeJavaDependencyArtifacts) {
      this.compileTimeJavaDependencyArtifactsBuilder.addTransitive(
          compileTimeJavaDependencyArtifacts);
      return this;
    }

    /**
     * Add the {@link JavaCompilationArgsProvider} for a dependency with export-like semantics; see
     * also {@link #addExports(JavaCompilationArgsProvider, ClasspathType)}.
     */
    public Builder addExports(JavaCompilationArgsProvider args) {
      return addExports(args, ClasspathType.BOTH);
    }

    /**
     * Add the {@link JavaCompilationArgsProvider} for a dependency with export-like semantics:
     * direct jars of the input are direct jars of the output.
     *
     * @param type of jars to collect; use {@link ClasspathType#RUNTIME_ONLY} for neverlink
     */
    public Builder addExports(JavaCompilationArgsProvider args, ClasspathType type) {
      return addArgs(args, type, true);
    }

    /**
     * Add the {@link JavaCompilationArgsProvider} for a dependency with dep-like semantics; see
     * also {@link #addDeps(JavaCompilationArgsProvider, ClasspathType)}.
     */
    public Builder addDeps(JavaCompilationArgsProvider args) {
      return addDeps(args, ClasspathType.BOTH);
    }

    /*
    * Add the {@link JavaCompilationArgsProvider} for a dependency with dep-like semantics:
    * direct jars of the input are <em>not</em> direct jars of the output.

    * @param type of jars to collect; use {@link ClasspathType#RUNTIME} for neverlink
    */
    public Builder addDeps(JavaCompilationArgsProvider args, ClasspathType type) {
      return addArgs(args, type, false);
    }

    /**
     * Includes the contents of another instance of {@link JavaCompilationArgsProvider}.
     *
     * @param args the {@link JavaCompilationArgsProvider} instance
     * @param type the classpath(s) to consider
     */
    private Builder addArgs(
        JavaCompilationArgsProvider args, ClasspathType type, boolean recursive) {
      if (!ClasspathType.RUNTIME_ONLY.equals(type)) {
        if (recursive) {
          directCompileTimeJarsBuilder.addTransitive(args.getDirectCompileTimeJars());
          directFullCompileTimeJarsBuilder.addTransitive(args.getDirectFullCompileTimeJars());
          compileTimeJavaDependencyArtifactsBuilder.addTransitive(
              args.getCompileTimeJavaDependencyArtifacts());
        }
        transitiveCompileTimeJarsBuilder.addTransitive(args.getTransitiveCompileTimeJars());
        transitiveFullCompileTimeJarsBuilder.addTransitive(args.getTransitiveFullCompileTimeJars());
      }
      if (!ClasspathType.COMPILE_ONLY.equals(type)) {
        runtimeJarsBuilder.addTransitive(args.getRuntimeJars());
      }
      return this;
    }

    /** Builds a {@link JavaCompilationArgsProvider}. */
    public JavaCompilationArgsProvider build() {
      if (runtimeJarsBuilder.isEmpty()
          && directCompileTimeJarsBuilder.isEmpty()
          && transitiveCompileTimeJarsBuilder.isEmpty()
          && directFullCompileTimeJarsBuilder.isEmpty()
          && transitiveFullCompileTimeJarsBuilder.isEmpty()
          && compileTimeJavaDependencyArtifactsBuilder.isEmpty()) {
        return EMPTY;
      }
      return create(
          runtimeJarsBuilder.build(),
          directCompileTimeJarsBuilder.build(),
          transitiveCompileTimeJarsBuilder.build(),
          directFullCompileTimeJarsBuilder.build(),
          transitiveFullCompileTimeJarsBuilder.build(),
          compileTimeJavaDependencyArtifactsBuilder.build());
    }
  }
}
