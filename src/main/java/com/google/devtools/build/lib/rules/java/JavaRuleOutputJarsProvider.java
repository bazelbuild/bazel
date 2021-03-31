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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaOutputApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaRuleOutputJarsProviderApi;
import java.util.Collection;
import java.util.Objects;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;

/** Provides information about jar files produced by a Java rule. */
@Immutable
@AutoCodec
public final class JavaRuleOutputJarsProvider
    implements TransitiveInfoProvider, JavaRuleOutputJarsProviderApi<JavaOutput> {

  public static final JavaRuleOutputJarsProvider EMPTY =
      new JavaRuleOutputJarsProvider(ImmutableList.<JavaOutput>of());

  /** A collection of artifacts associated with a jar output. */
  @AutoValue
  @Immutable
  @AutoCodec
  public abstract static class JavaOutput implements JavaOutputApi<Artifact> {
    @Override
    public boolean isImmutable() {
      return true; // immutable and Starlark-hashable
    }

    @Override
    public abstract Artifact getClassJar();

    @Nullable
    @Override
    public abstract Artifact getCompileJar();

    @Nullable
    @Deprecated
    @Override
    public Artifact getIJar() {
      return getCompileJar();
    }

    @Nullable
    @Override
    public abstract Artifact getCompileJdeps();

    @Nullable
    @Override
    public abstract Artifact getGeneratedClassJar();

    @Nullable
    @Override
    public abstract Artifact getGeneratedSourceJar();

    @Nullable
    @Override
    public abstract Artifact getNativeHeadersJar();

    @Nullable
    @Override
    public abstract Artifact getManifestProto();

    @Nullable
    @Override
    public abstract Artifact getJdeps();

    @Nullable
    @Deprecated
    @Override
    public Artifact getSrcJar() {
      return Iterables.getOnlyElement(getSourceJars(), null);
    }

    @Nullable
    @Override
    public Sequence<Artifact> getSrcJarsStarlark() {
      return StarlarkList.immutableCopyOf(getSourceJars());
    }

    /** A list of sources archive files. */
    public abstract ImmutableList<Artifact> getSourceJars();

    @AutoCodec.Instantiator
    public static JavaOutput create(
        Artifact classJar,
        @Nullable Artifact compileJar,
        @Nullable Artifact compileJdeps,
        @Nullable Artifact generatedClassJar,
        @Nullable Artifact generatedSourceJar,
        @Nullable Artifact nativeHeadersJar,
        @Nullable Artifact manifestProto,
        @Nullable Artifact jdeps,
        ImmutableList<Artifact> sourceJars) {
      return builder()
          .setClassJar(classJar)
          .setCompileJar(compileJar)
          .setCompileJdeps(compileJdeps)
          .setGeneratedClassJar(generatedClassJar)
          .setGeneratedSourceJar(generatedSourceJar)
          .setNativeHeadersJar(nativeHeadersJar)
          .setManifestProto(manifestProto)
          .setJdeps(jdeps)
          .addSourceJars(sourceJars)
          .build();
    }

    /** Builder for OutputJar. */
    @AutoValue.Builder
    public abstract static class Builder {

      public abstract Builder setClassJar(Artifact value);

      public abstract Builder setCompileJar(Artifact value);

      public abstract Builder setCompileJdeps(Artifact value);

      public abstract Builder setGeneratedClassJar(Artifact value);

      public abstract Builder setGeneratedSourceJar(Artifact value);

      public abstract Builder setNativeHeadersJar(Artifact value);

      public abstract Builder setManifestProto(Artifact value);

      public abstract Builder setJdeps(Artifact value);

      public abstract Builder setSourceJars(Iterable<Artifact> value);

      abstract ImmutableList.Builder<Artifact> sourceJarsBuilder();

      public Builder addSourceJar(@Nullable Artifact value) {
        if (value != null) {
          sourceJarsBuilder().add(value);
        }
        return this;
      }

      public Builder addSourceJars(Iterable<Artifact> values) {
        sourceJarsBuilder().addAll(values);
        return this;
      }

      /** Populates the builder with outputs from {@link JavaCompileOutputs}. */
      public Builder fromJavaCompileOutputs(JavaCompileOutputs<Artifact> value) {
        setClassJar(value.output());
        setJdeps(value.depsProto());
        setGeneratedClassJar(value.genClass());
        setGeneratedSourceJar(value.genSource());
        setNativeHeadersJar(value.nativeHeader());
        setManifestProto(value.manifestProto());
        return this;
      }

      public abstract JavaOutput build();
    }

    public static Builder builder() {
      return new AutoValue_JavaRuleOutputJarsProvider_JavaOutput.Builder();
    }
  }

  final ImmutableList<JavaOutput> javaOutputs;

  private JavaRuleOutputJarsProvider(ImmutableList<JavaOutput> javaOutputs) {
    this.javaOutputs = javaOutputs;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static JavaRuleOutputJarsProvider create(ImmutableList<JavaOutput> javaOutputs) {
    if (javaOutputs.isEmpty()) {
      return EMPTY;
    }
    return new JavaRuleOutputJarsProvider(javaOutputs);
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public ImmutableList<JavaOutput> getJavaOutputs() {
    return javaOutputs;
  }

  /** Collects all class output jars from {@link #javaOutputs} */
  public Iterable<Artifact> getAllClassOutputJars() {
    return javaOutputs.stream().map(JavaOutput::getClassJar).collect(Collectors.toList());
  }

  /** Collects all source output jars from {@link #javaOutputs} */
  public ImmutableList<Artifact> getAllSrcOutputJars() {
    return javaOutputs.stream()
        .map(JavaOutput::getSourceJars)
        .flatMap(ImmutableList::stream)
        .collect(toImmutableList());
  }

  @Nullable
  @Override
  @Deprecated
  public Artifact getJdeps() {
    ImmutableList<Artifact> jdeps =
        javaOutputs.stream()
            .map(JavaOutput::getJdeps)
            .filter(Objects::nonNull)
            .collect(toImmutableList());
    return jdeps.size() == 1 ? jdeps.get(0) : null;
  }

  @Nullable
  @Override
  @Deprecated
  public Artifact getNativeHeaders() {
    ImmutableList<Artifact> nativeHeaders =
        javaOutputs.stream()
            .map(JavaOutput::getNativeHeadersJar)
            .filter(Objects::nonNull)
            .collect(toImmutableList());
    return nativeHeaders.size() == 1 ? nativeHeaders.get(0) : null;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static JavaRuleOutputJarsProvider merge(Collection<JavaRuleOutputJarsProvider> providers) {
    Builder builder = new Builder();
    for (JavaRuleOutputJarsProvider provider : providers) {
      builder.addJavaOutput(provider.getJavaOutputs());
    }
    return builder.build();
  }

  /** Builder for {@link JavaRuleOutputJarsProvider}. */
  public static class Builder {
    private final ImmutableList.Builder<JavaOutput> javaOutputs = ImmutableList.builder();

    public Builder addJavaOutput(JavaOutput javaOutput) {
      javaOutputs.add(javaOutput);
      return this;
    }

    public Builder addJavaOutput(Iterable<JavaOutput> javaOutputs) {
      this.javaOutputs.addAll(javaOutputs);
      return this;
    }

    public JavaRuleOutputJarsProvider build() {
      return new JavaRuleOutputJarsProvider(javaOutputs.build());
    }
  }
}
