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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaRuleOutputJarsProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.OutputJarApi;
import java.util.Collection;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;

/** Provides information about jar files produced by a Java rule. */
@Immutable
@AutoCodec
public final class JavaRuleOutputJarsProvider
    implements TransitiveInfoProvider, JavaRuleOutputJarsProviderApi<OutputJar> {

  public static final JavaRuleOutputJarsProvider EMPTY =
      new JavaRuleOutputJarsProvider(
          ImmutableList.<OutputJar>of(), /* jdeps= */ null, /* nativeHeaders= */ null);

  /** A collection of artifacts associated with a jar output. */
  @Immutable
  @AutoCodec
  public static class OutputJar implements OutputJarApi<Artifact> {
    private final Artifact classJar;
    @Nullable private final Artifact iJar;
    @Nullable private final Artifact manifestProto;
    @Nullable private final ImmutableList<Artifact> srcJars;

    public OutputJar(
        Artifact classJar,
        @Nullable Artifact iJar,
        @Nullable Artifact manifestProto,
        @Nullable Iterable<Artifact> srcJars) {
      this.classJar = classJar;
      this.iJar = iJar;
      this.manifestProto = manifestProto;
      this.srcJars = ImmutableList.copyOf(srcJars);
    }

    @Override
    public boolean isImmutable() {
      return true; // immutable and Starlark-hashable
    }

    @Override
    public Artifact getClassJar() {
      return classJar;
    }

    @Nullable
    @Override
    public Artifact getIJar() {
      return iJar;
    }

    @Nullable
    @Override
    public Artifact getManifestProto() {
      return manifestProto;
    }

    @Nullable
    @Override
    public Artifact getSrcJar() {
      return Iterables.getOnlyElement(srcJars, null);
    }

    @Nullable
    @Override
    public Sequence<Artifact> getSrcJarsStarlark() {
      return StarlarkList.immutableCopyOf(srcJars);
    }

    public Iterable<Artifact> getSrcJars() {
      return srcJars;
    }
  }

  final ImmutableList<OutputJar> outputJars;
  @Nullable final Artifact jdeps;
  /** An archive of native header files. */
  @Nullable final Artifact nativeHeaders;

  private JavaRuleOutputJarsProvider(
      ImmutableList<OutputJar> outputJars,
      @Nullable Artifact jdeps,
      @Nullable Artifact nativeHeaders) {
    this.outputJars = outputJars;
    this.jdeps = jdeps;
    this.nativeHeaders = nativeHeaders;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static JavaRuleOutputJarsProvider create(
      ImmutableList<OutputJar> outputJars,
      @Nullable Artifact jdeps,
      @Nullable Artifact nativeHeaders) {
    if (outputJars.isEmpty() && jdeps == null && nativeHeaders == null) {
      return EMPTY;
    }
    return new JavaRuleOutputJarsProvider(outputJars, jdeps, nativeHeaders);
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public ImmutableList<OutputJar> getOutputJars() {
    return outputJars;
  }

  /** Collects all class output jars from {@link #outputJars} */
  public Iterable<Artifact> getAllClassOutputJars() {
    return outputJars.stream().map(OutputJar::getClassJar).collect(Collectors.toList());
  }

  /** Collects all source output jars from {@link #outputJars} */
  public Iterable<Artifact> getAllSrcOutputJars() {
    return outputJars.stream()
        .map(OutputJar::getSrcJars)
        .reduce(ImmutableList.of(), Iterables::concat);
  }

  @Nullable
  @Override
  public Artifact getJdeps() {
    return jdeps;
  }

  @Nullable
  @Override
  public Artifact getNativeHeaders() {
    return nativeHeaders;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static JavaRuleOutputJarsProvider merge(Collection<JavaRuleOutputJarsProvider> providers) {
    Builder builder = new Builder();
    for (JavaRuleOutputJarsProvider provider : providers) {
      builder.addOutputJars(provider.getOutputJars());
    }
    return builder.build();
  }

  /** Builder for {@link JavaRuleOutputJarsProvider}. */
  public static class Builder {
    private final ImmutableList.Builder<OutputJar> outputJars = ImmutableList.builder();
    private Artifact jdeps;
    private Artifact nativeHeaders;

    public Builder addOutputJar(
        Artifact classJar,
        @Nullable Artifact iJar,
        @Nullable Artifact manifestProto,
        @Nullable ImmutableList<Artifact> sourceJars) {
      Preconditions.checkState(classJar != null);
      outputJars.add(new OutputJar(classJar, iJar, manifestProto, sourceJars));
      return this;
    }

    public Builder addOutputJar(OutputJar outputJar) {
      outputJars.add(outputJar);
      return this;
    }

    public Builder addOutputJars(Iterable<OutputJar> outputJars) {
      this.outputJars.addAll(outputJars);
      return this;
    }

    public Builder setJdeps(Artifact jdeps) {
      this.jdeps = jdeps;
      return this;
    }

    public Builder setNativeHeaders(Artifact nativeHeaders) {
      this.nativeHeaders = requireNonNull(nativeHeaders);
      return this;
    }

    public JavaRuleOutputJarsProvider build() {
      return new JavaRuleOutputJarsProvider(outputJars.build(), jdeps, nativeHeaders);
    }
  }
}
