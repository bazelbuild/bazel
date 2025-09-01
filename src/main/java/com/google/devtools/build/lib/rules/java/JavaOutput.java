// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoBuilder;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaOutputApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A collection of artifacts associated with a jar output.
 *
 * @param sourceJars A {@link NestedSet} of sources archive files.
 */
@Immutable
@AutoCodec
public record JavaOutput(
    @Override Artifact classJar,
    @Nullable @Override Artifact compileJar,
    @Nullable @Override Artifact headerCompilationJar,
    @Nullable @Override Artifact compileJdeps,
    @Nullable @Override Artifact generatedClassJar,
    @Nullable @Override Artifact generatedSourceJar,
    @Nullable @Override Artifact nativeHeadersJar,
    @Nullable @Override Artifact manifestProto,
    @Nullable @Override Artifact jdeps,
    NestedSet<Artifact> sourceJars)
    implements JavaOutputApi<Artifact> {
  public JavaOutput {
    requireNonNull(classJar, "classJar");
    requireNonNull(sourceJars, "sourceJars");
  }

  @Override
  public Artifact getClassJar() {
    return classJar();
  }

  @Nullable
  @Override
  public Artifact getCompileJar() {
    return compileJar();
  }

  @Nullable
  @Override
  public Artifact getHeaderCompilationJar() {
    return headerCompilationJar;
  }

  @Nullable
  @Override
  public Artifact getCompileJdeps() {
    return compileJdeps();
  }

  @Nullable
  @Override
  public Artifact getGeneratedClassJar() {
    return generatedClassJar();
  }

  @Nullable
  @Override
  public Artifact getGeneratedSourceJar() {
    return generatedSourceJar();
  }

  @Nullable
  @Override
  public Artifact getNativeHeadersJar() {
    return nativeHeadersJar();
  }

  @Nullable
  @Override
  public Artifact getManifestProto() {
    return manifestProto();
  }

  @Nullable
  @Override
  public Artifact getJdeps() {
    return jdeps();
  }

  /**
   * Translates a collection of {@link JavaOutput} for use in native code.
   *
   * @param outputs the collection of translate
   * @return an immutable list of {@link JavaOutput} instances
   * @throws EvalException if there were errors reading fields from the {@code Starlark} object
   * @throws RuleErrorException if any item in the supplied collection is not a valid {@link
   *     JavaOutput}
   */
  @VisibleForTesting
  public static ImmutableList<JavaOutput> wrapSequence(Collection<?> outputs)
      throws EvalException, RuleErrorException {
    ImmutableList.Builder<JavaOutput> result = ImmutableList.builder();
    for (Object info : outputs) {
      if (info instanceof JavaOutput javaOutput) {
        result.add(javaOutput);
      } else if (info instanceof StructImpl structImpl) {
        result.add(fromStarlarkJavaOutput(structImpl));
      } else {
        throw new RuleErrorException("expected JavaOutput, got: " + Starlark.type(info));
      }
    }
    return result.build();
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Nullable
  @Deprecated
  @Override
  public Artifact getIJar() {
    return compileJar();
  }

  @Nullable
  @Deprecated
  @Override
  public Artifact getSrcJar() {
    return Iterables.getOnlyElement(getSourceJarsAsList(), null);
  }

  public ImmutableList<Artifact> getSourceJarsAsList() {
    return sourceJars().toList();
  }

  @Nullable
  @Override
  public Depset getSrcJarsStarlark(StarlarkSemantics semantics) {
    return Depset.of(Artifact.class, sourceJars());
  }

  public static JavaOutput fromStarlarkJavaOutput(StructImpl struct) throws EvalException {
    NestedSet<Artifact> sourceJars;
    Object starlarkSourceJars = struct.getValue("source_jars");
    if (starlarkSourceJars == Starlark.NONE || starlarkSourceJars instanceof Depset) {
      sourceJars = Depset.noneableCast(starlarkSourceJars, Artifact.class, "source_jars");
    } else {
      sourceJars =
          NestedSetBuilder.wrap(
              Order.STABLE_ORDER, Sequence.cast(starlarkSourceJars, Artifact.class, "source_jars"));
    }
    return JavaOutput.builder()
        .setClassJar(nullIfNone(struct.getValue("class_jar"), Artifact.class))
        .setCompileJar(nullIfNone(struct.getValue("compile_jar"), Artifact.class))
        .setHeaderCompilationJar(
            nullIfNone(struct.getValue("header_compilation_jar"), Artifact.class))
        .setCompileJdeps(nullIfNone(struct.getValue("compile_jdeps"), Artifact.class))
        .setGeneratedClassJar(nullIfNone(struct.getValue("generated_class_jar"), Artifact.class))
        .setGeneratedSourceJar(nullIfNone(struct.getValue("generated_source_jar"), Artifact.class))
        .setNativeHeadersJar(nullIfNone(struct.getValue("native_headers_jar"), Artifact.class))
        .setManifestProto(nullIfNone(struct.getValue("manifest_proto"), Artifact.class))
        .setJdeps(nullIfNone(struct.getValue("jdeps"), Artifact.class))
        .addSourceJars(sourceJars)
        .build();
  }

  @Nullable
  static <T> T nullIfNone(Object object, Class<T> type) {
    return object != Starlark.NONE ? type.cast(object) : null;
  }

  /** Builder for OutputJar. */
  @AutoBuilder
  public abstract static class Builder {
    private final NestedSetBuilder<Artifact> sourceJarsBuilder = NestedSetBuilder.stableOrder();

    public abstract Builder setClassJar(Artifact value);

    public abstract Builder setCompileJar(Artifact value);

    public abstract Builder setHeaderCompilationJar(Artifact value);

    public abstract Builder setCompileJdeps(Artifact value);

    public abstract Builder setGeneratedClassJar(Artifact value);

    public abstract Builder setGeneratedSourceJar(Artifact value);

    public abstract Builder setNativeHeadersJar(Artifact value);

    public abstract Builder setManifestProto(Artifact value);

    public abstract Builder setJdeps(Artifact value);

    @CanIgnoreReturnValue
    abstract Builder setSourceJars(NestedSet<Artifact> value);

    public Builder addSourceJar(@Nullable Artifact value) {
      if (value != null) {
        sourceJarsBuilder.add(value);
      }
      return this;
    }

    public Builder addSourceJars(NestedSet<Artifact> values) {
      sourceJarsBuilder.addTransitive(values);
      return this;
    }

    /** Populates the builder with outputs from {@link JavaCompileOutputs}. */
    public Builder fromJavaCompileOutputs(JavaCompileOutputs<Artifact> value) {
      return fromJavaCompileOutputs(value, true);
    }

    @CanIgnoreReturnValue
    public Builder fromJavaCompileOutputs(
        JavaCompileOutputs<Artifact> value, boolean includeJdeps) {
      setClassJar(value.output());
      if (includeJdeps) {
        setJdeps(value.depsProto());
      }
      setGeneratedClassJar(value.genClass());
      setGeneratedSourceJar(value.genSource());
      setNativeHeadersJar(value.nativeHeader());
      setManifestProto(value.manifestProto());
      return this;
    }

    abstract JavaOutput autoBuild();

    public JavaOutput build() {
      setSourceJars(sourceJarsBuilder.build());
      return autoBuild();
    }
  }

  public static Builder builder() {
    return new AutoBuilder_JavaOutput_Builder();
  }
}
