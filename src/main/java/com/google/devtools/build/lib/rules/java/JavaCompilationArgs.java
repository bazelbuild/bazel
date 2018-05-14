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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A container of Java compilation artifacts. */
@AutoValue
@AutoCodec
public abstract class JavaCompilationArgs {
  // TODO(bazel-team): It would be desirable to use LinkOrderNestedSet here so that
  // parents-before-deps is preserved for graphs that are not trees. However, the legacy
  // JavaLibraryCollector implemented naive link ordering and many targets in the
  // depot depend on the consistency of left-to-right ordering that is not provided by
  // LinkOrderNestedSet. They simply list their local dependencies before
  // other targets that may use conflicting dependencies, and the local deps
  // appear earlier on the classpath, as desired. Behavior of LinkOrderNestedSet
  // can be very unintuitive in case of conflicting orders, because the order is
  // decided by the rightmost branch in such cases. For example, if A depends on {junit4,
  // B}, B depends on {C, D}, C depends on {junit3}, and D depends on {junit4},
  // the classpath of A will have junit3 before junit4.

  @AutoCodec
  public static final JavaCompilationArgs EMPTY_ARGS =
      JavaCompilationArgs.create(
          NestedSetBuilder.<Artifact>create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.<Artifact>create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.<Artifact>create(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.<Artifact>create(Order.NAIVE_LINK_ORDER));

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static JavaCompilationArgs create(
      NestedSet<Artifact> runtimeJars,
      NestedSet<Artifact> compileTimeJars,
      NestedSet<Artifact> fullCompileTimeJars,
      NestedSet<Artifact> instrumentationMetadata) {
    return new AutoValue_JavaCompilationArgs(
        runtimeJars, compileTimeJars, fullCompileTimeJars, instrumentationMetadata);
  }

  /** Returns transitive runtime jars. */
  public abstract NestedSet<Artifact> getRuntimeJars();

  /** Returns transitive compile-time jars. */
  public abstract NestedSet<Artifact> getCompileTimeJars();

  /**
   * Returns transitive full jars for compilation time: the full jars for every interface jar and
   * all the full jars on the compiletime classpath.
   */
  public abstract NestedSet<Artifact> getFullCompileTimeJars();

  /** Returns transitive instrumentation metadata jars. */
  public abstract NestedSet<Artifact> getInstrumentationMetadata();

  /**
   * Returns a new builder instance.
   */
  public static final Builder builder() {
    return new Builder();
  }

  /**
   * Builder for {@link JavaCompilationArgs}.
   */
  public static final class Builder {
    private final NestedSetBuilder<Artifact> runtimeJarsBuilder =
        NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> compileTimeJarsBuilder =
        NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> fullCompileTimeJarsBuilder =
        NestedSetBuilder.naiveLinkOrder();
    private final NestedSetBuilder<Artifact> instrumentationMetadataBuilder =
        NestedSetBuilder.naiveLinkOrder();

    /**
     * Use {@code TransitiveJavaCompilationArgs#builder()} to instantiate the builder.
     */
    private Builder() {
    }

    public static Builder copyOf(Builder builder){
      Builder result = new Builder();
      result.addTransitiveRuntimeJars(builder.runtimeJarsBuilder.build());
      result.addTransitiveCompileTimeJars(builder.compileTimeJarsBuilder.build());
      result.addTransitiveFullCompileTimeJars(builder.fullCompileTimeJarsBuilder.build());
      result.addInstrumentationMetadata(builder.instrumentationMetadataBuilder.build());
      return result;
    }

    /**
     * Legacy method for dealing with objects which construct
     * {@link JavaCompilationArtifacts} objects.
     */
    // TODO(bazel-team): Remove when we get rid of JavaCompilationArtifacts.
    public Builder merge(JavaCompilationArtifacts other, boolean isNeverLink) {
      if (!isNeverLink) {
        addRuntimeJars(other.getRuntimeJars());
      }
      addCompileTimeJars(other.getCompileTimeJars());
      addFullCompileTimeJars(other.getFullCompileTimeJars());
      addInstrumentationMetadata(other.getInstrumentationMetadata());
      return this;
    }

    /**
     * Legacy method for dealing with objects which construct
     * {@link JavaCompilationArtifacts} objects.
     */
    public Builder merge(JavaCompilationArtifacts other) {
      return merge(other, false);
    }

    public Builder addRuntimeJar(Artifact runtimeJar) {
      this.runtimeJarsBuilder.add(runtimeJar);
      return this;
    }

    public Builder addRuntimeJars(Iterable<Artifact> runtimeJars) {
      this.runtimeJarsBuilder.addAll(runtimeJars);
      return this;
    }

    public Builder addTransitiveRuntimeJars(NestedSet<Artifact> runtimeJars) {
      this.runtimeJarsBuilder.addTransitive(runtimeJars);
      return this;
    }

    public Builder addCompileTimeJarAsFullJar(Artifact compileTimeJar) {
      this.compileTimeJarsBuilder.add(compileTimeJar);
      this.fullCompileTimeJarsBuilder.add(compileTimeJar);
      return this;
    }

    public Builder addCompileTimeJars(Iterable<Artifact> compileTimeJars) {
      this.compileTimeJarsBuilder.addAll(compileTimeJars);
      return this;
    }

    public Builder addCompileTimeJar(Artifact compileTimeJar) {
      this.compileTimeJarsBuilder.add(compileTimeJar);
      return this;
    }

    public Builder addFullCompileTimeJars(Iterable<Artifact> fullCompileTimeJars) {
      this.fullCompileTimeJarsBuilder.addAll(fullCompileTimeJars);
      return this;
    }

    public Builder addFullCompileTimeJar(Artifact fullCompileTimeJar) {
      this.fullCompileTimeJarsBuilder.add(fullCompileTimeJar);
      return this;
    }

    public Builder addTransitiveCompileTimeJars(NestedSet<Artifact> compileTimeJars) {
      this.compileTimeJarsBuilder.addTransitive(compileTimeJars);
      return this;
    }

    public Builder addTransitiveFullCompileTimeJars(NestedSet<Artifact> fullCompileTimeJars) {
      this.fullCompileTimeJarsBuilder.addTransitive(fullCompileTimeJars);
      return this;
    }

    public Builder addInstrumentationMetadata(Artifact instrumentationMetadata) {
      this.instrumentationMetadataBuilder.add(instrumentationMetadata);
      return this;
    }

    public Builder addInstrumentationMetadata(Iterable<Artifact> instrumentationMetadata) {
      this.instrumentationMetadataBuilder.addAll(instrumentationMetadata);
      return this;
    }

    public Builder addTransitiveCompilationArgs(
        JavaCompilationArgsProvider dep, boolean recursive, ClasspathType type) {
      JavaCompilationArgs args =
          recursive ? dep.getRecursiveJavaCompilationArgs() : dep.getJavaCompilationArgs();
      addTransitiveArgs(args, type);
      return this;
    }

    public Builder addTransitiveCompilationArgs(
        Iterable<JavaCompilationArgsProvider> args, boolean recursive, ClasspathType type) {
      for (JavaCompilationArgsProvider provider : args) {
        addTransitiveCompilationArgs(provider, recursive, type);
      }
      return this;
    }

    /**
     * Includes the contents of another instance of {@link JavaCompilationArgs}.
     *
     * @param args the {@link JavaCompilationArgs} instance
     * @param type the classpath(s) to consider
     */
    public Builder addTransitiveArgs(JavaCompilationArgs args, ClasspathType type) {
      if (!ClasspathType.RUNTIME_ONLY.equals(type)) {
        compileTimeJarsBuilder.addTransitive(args.getCompileTimeJars());
        fullCompileTimeJarsBuilder.addTransitive(args.getFullCompileTimeJars());
      }
      if (!ClasspathType.COMPILE_ONLY.equals(type)) {
        runtimeJarsBuilder.addTransitive(args.getRuntimeJars());
      }
      instrumentationMetadataBuilder.addTransitive(
          args.getInstrumentationMetadata());
      return this;
    }

    /**
     * Builds a {@link JavaCompilationArgs} object.
     */
    public JavaCompilationArgs build() {
      if (runtimeJarsBuilder.isEmpty()
          && compileTimeJarsBuilder.isEmpty()
          && fullCompileTimeJarsBuilder.isEmpty()
          && instrumentationMetadataBuilder.isEmpty()) {
        return EMPTY_ARGS;
      }
      return JavaCompilationArgs.create(
          runtimeJarsBuilder.build(),
          compileTimeJarsBuilder.build(),
          fullCompileTimeJarsBuilder.build(),
          instrumentationMetadataBuilder.build());
    }
  }

  /**
   *  Enum to specify transitive compilation args traversal
   */
  public static enum ClasspathType {
      /* treat the same for compile time and runtime */
      BOTH,

      /* Only include on compile classpath */
      COMPILE_ONLY,

      /* Only include on runtime classpath */
      RUNTIME_ONLY;
  }

  JavaCompilationArgs() {}
}
