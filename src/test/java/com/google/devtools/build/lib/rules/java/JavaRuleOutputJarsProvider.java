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
import static java.util.Objects.requireNonNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.java.JavaInfo.JavaInfoInternalProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaRuleOutputJarsProviderApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.Objects;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;

/** Provides information about jar files produced by a Java rule. */
@Immutable
@AutoCodec
public record JavaRuleOutputJarsProvider(@Override ImmutableList<JavaOutput> javaOutputs)
    implements JavaInfoInternalProvider, JavaRuleOutputJarsProviderApi<JavaOutput> {
  public JavaRuleOutputJarsProvider {
    requireNonNull(javaOutputs, "javaOutputs");
  }

  @Override
  public ImmutableList<JavaOutput> getJavaOutputs() {
    return javaOutputs();
  }

  @SerializationConstant
  public static final JavaRuleOutputJarsProvider EMPTY =
      new JavaRuleOutputJarsProvider(ImmutableList.<JavaOutput>of());

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /** Collects all class output jars from {@link #getJavaOutputs} */
  public Iterable<Artifact> getAllClassOutputJars() {
    return javaOutputs().stream().map(JavaOutput::classJar).collect(Collectors.toList());
  }

  /** Collects all source output jars from {@link #getJavaOutputs} */
  public ImmutableList<Artifact> getAllSrcOutputJars() {
    return javaOutputs().stream()
        .map(JavaOutput::getSourceJarsAsList)
        .flatMap(ImmutableList::stream)
        .collect(toImmutableList());
  }

  @Nullable
  @Override
  @Deprecated
  public Artifact getJdeps() {
    ImmutableList<Artifact> jdeps =
        javaOutputs().stream()
            .map(JavaOutput::jdeps)
            .filter(Objects::nonNull)
            .collect(toImmutableList());
    return jdeps.size() == 1 ? jdeps.get(0) : null;
  }

  @Nullable
  @Override
  @Deprecated
  public Artifact getNativeHeaders() {
    ImmutableList<Artifact> nativeHeaders =
        javaOutputs().stream()
            .map(JavaOutput::nativeHeadersJar)
            .filter(Objects::nonNull)
            .collect(toImmutableList());
    return nativeHeaders.size() == 1 ? nativeHeaders.get(0) : null;
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link JavaRuleOutputJarsProvider}. */
  public static class Builder {
    // CompactHashSet preserves insertion order here since we never perform any removals
    private final CompactHashSet<JavaOutput> javaOutputs = CompactHashSet.create();

    @CanIgnoreReturnValue
    public Builder addJavaOutput(JavaOutput javaOutput) {
      javaOutputs.add(javaOutput);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addJavaOutput(Collection<JavaOutput> javaOutputs) {
      this.javaOutputs.addAll(javaOutputs);
      return this;
    }

    public JavaRuleOutputJarsProvider build() {
      return new JavaRuleOutputJarsProvider(ImmutableList.copyOf(javaOutputs));
    }
  }

  /**
   * Translates the {@code outputs} field of a {@link JavaInfo} instance into a native {@link
   * JavaRuleOutputJarsProvider} instance.
   *
   * <p>This method first attempts to transform the {@code outputs} field of the supplied {@code
   * JavaInfo}. If this is not present (for example, in Bazel), it attempts to create the result
   * from the {@code java_outputs} field instead.
   *
   * @param javaInfo the {@link JavaInfo} instance
   * @return a {@link JavaRuleOutputJarsProvider} instance
   * @throws EvalException if there are any errors accessing Starlark values
   * @throws RuleErrorException if any of the {@code output} instances are of incompatible type
   */
  static JavaRuleOutputJarsProvider fromStarlarkJavaInfo(StructImpl javaInfo)
      throws EvalException, RuleErrorException {
    Object outputs = javaInfo.getValue("outputs");
    if (outputs == null) {
      return JavaRuleOutputJarsProvider.builder()
          .addJavaOutput(
              JavaOutput.wrapSequence(
                  Sequence.cast(javaInfo.getValue("java_outputs"), Objects.class, "java_outputs")))
          .build();
    } else {
      return fromStarlark(outputs);
    }
  }

  /**
   * Translates the supplied object into a {@link JavaRuleOutputJarsProvider} instance.
   *
   * @param obj the object to translate
   * @return a {@link JavaRuleOutputJarsProvider} instance, or null if the supplied object was null
   *     or {@link Starlark#NONE}
   * @throws EvalException if there were any errors reading fields from the supplied object
   * @throws RuleErrorException if the supplied object is not a {@link JavaRuleOutputJarsProvider}
   */
  @VisibleForTesting
  public static JavaRuleOutputJarsProvider fromStarlark(Object obj)
      throws EvalException, RuleErrorException {
    if (obj == Starlark.NONE) {
      return JavaRuleOutputJarsProvider.EMPTY;
    } else if (obj instanceof JavaRuleOutputJarsProvider javaRuleOutputJarsProvider) {
      return javaRuleOutputJarsProvider;
    } else if (obj instanceof StructImpl) {
      return JavaRuleOutputJarsProvider.builder()
          .addJavaOutput(
              JavaOutput.wrapSequence(
                  Sequence.cast(((StructImpl) obj).getValue("jars"), Object.class, "jars")))
          .build();
    } else {
      throw new RuleErrorException(
          "expected JavaRuleOutputJarsProvider, got: " + Starlark.type(obj));
    }
  }
}
