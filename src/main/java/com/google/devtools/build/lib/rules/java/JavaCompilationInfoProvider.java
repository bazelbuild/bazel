// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.java.JavaInfo.JavaInfoInternalProvider;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaCompilationInfoProviderApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;

/**
 * A class that provides compilation information in Java rules, for perusal of aspects and tools.
 */
@Immutable
@AutoValue
public abstract class JavaCompilationInfoProvider
    implements JavaInfoInternalProvider, JavaCompilationInfoProviderApi<Artifact> {

  /**
   * Transforms the {@code compilation_info} field from a {@link JavaInfo} into a native instance.
   *
   * @param javaInfo A {@link JavaInfo} instance.
   * @return a {@link JavaCompilationInfoProvider} instance or {@code null} if the {@code
   *     compilation_info} field is not present in the supplied {@code javaInfo}
   * @throws RuleErrorException if the {@code compilation_info} is of an incompatible type
   * @throws EvalException if there are any errors accessing Starlark values
   */
  @Nullable
  static JavaCompilationInfoProvider fromStarlarkJavaInfo(StructImpl javaInfo)
      throws RuleErrorException, EvalException {
    Object value = javaInfo.getValue("compilation_info");
    return fromStarlarkCompilationInfo(value);
  }

  /**
   * Translates an instance of {@link JavaCompilationInfoProvider} for use in native code.
   *
   * @param value The object to translate
   * @return a {@link JavaCompilationInfoProvider} instance, or null if the supplied value is null
   *     or {@link Starlark#NONE}
   * @throws EvalException if there are errors reading any fields from the {@link StructImpl}
   * @throws RuleErrorException if the supplied value is not compatible with {@link
   *     JavaCompilationInfoProvider}
   */
  @Nullable
  @VisibleForTesting
  public static JavaCompilationInfoProvider fromStarlarkCompilationInfo(Object value)
      throws EvalException, RuleErrorException {
    if (value == null || value == Starlark.NONE) {
      return null;
    } else if (value instanceof JavaCompilationInfoProvider) {
      return (JavaCompilationInfoProvider) value;
    } else if (value instanceof StructImpl) {
      StructImpl info = (StructImpl) value;
      Builder builder =
          new Builder()
              .setJavacOpts(
                  Sequence.cast(info.getValue("javac_options"), String.class, "javac_options")
                      .getImmutableList())
              .setBootClasspath(
                  BootClassPathInfo.create(
                      NestedSetBuilder.wrap(
                          Order.NAIVE_LINK_ORDER,
                          Sequence.noneableCast(
                              info.getValue("boot_classpath"), Artifact.class, "boot_classpath"))));
      Object runtimeClasspath = info.getValue("runtime_classpath");
      if (runtimeClasspath != null) {
        builder.setRuntimeClasspath(
            Depset.noneableCast(runtimeClasspath, Artifact.class, "runtime_classpath"));
      }
      Object compilationClasspath = info.getValue("compilation_classpath");
      if (compilationClasspath != null) {
        builder.setCompilationClasspath(
            Depset.noneableCast(compilationClasspath, Artifact.class, "compilation_classpath"));
      }
      return builder.build();
    }
    throw new RuleErrorException("expected java_compilation_info, got: " + Starlark.type(value));
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /** Builder for {@link JavaCompilationInfoProvider}. */
  public static class Builder {
    private ImmutableList<String> javacOpts;
    private NestedSet<Artifact> runtimeClasspath;
    private NestedSet<Artifact> compilationClasspath;
    private BootClassPathInfo bootClasspath = BootClassPathInfo.empty();

    @CanIgnoreReturnValue
    public Builder setJavacOpts(ImmutableList<String> javacOpts) {
      this.javacOpts = javacOpts;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setRuntimeClasspath(@Nullable NestedSet<Artifact> runtimeClasspath) {
      this.runtimeClasspath = runtimeClasspath;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setCompilationClasspath(@Nullable NestedSet<Artifact> compilationClasspath) {
      this.compilationClasspath = compilationClasspath;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setBootClasspath(BootClassPathInfo bootClasspath) {
      this.bootClasspath = Preconditions.checkNotNull(bootClasspath);
      return this;
    }

    public JavaCompilationInfoProvider build() {
      return new AutoValue_JavaCompilationInfoProvider(
          JavaCompilationHelper.internJavacOpts(javacOpts),
          runtimeClasspath,
          compilationClasspath,
          bootClasspath.bootclasspath());
    }
  }

  @Override
  @Nullable
  public abstract ImmutableList<String> getJavacOpts();

  @Nullable
  public abstract NestedSet<Artifact> runtimeClasspath();

  @Override
  @Nullable
  public Depset /*<Artifact>*/ getRuntimeClasspath() {
    return runtimeClasspath() == null ? null : Depset.of(Artifact.class, runtimeClasspath());
  }

  @Nullable
  public abstract NestedSet<Artifact> compilationClasspath();

  @Override
  @Nullable
  public Depset /*<Artifact>*/ getCompilationClasspath() {
    return compilationClasspath() == null
        ? null
        : Depset.of(Artifact.class, compilationClasspath());
  }

  @Override
  public ImmutableList<Artifact> getBootClasspathList() {
    return bootClasspath().toList();
  }

  public abstract NestedSet<Artifact> bootClasspath();

  /*
   * Underrides the @Autovalue implementation.
   * We shouldn't be doing this, but this is necessary to allow Starlark-constructed instances to
   * be compared with natively constructed instances. The difference arises only because of the
   * boot classpath. The Starlark API returns a list, while we store a NestedSet in
   * native for efficiency. When we reconstruct a native instance from a Starlark one, the list is
   * wrapped in a new NestedSet instance. Since NestedSet equality relies on
   * reference-equality, here, we perform a NestedSet#shallowEquals only for the bootClasspath to
   * verify the contents are the same.
   * Note: this is temporary, and is required only while JavaCompilationInfoProvider is still
   * constructed in native code. Once the migration to Starlark is complete, this will be deleted as
   * this class will no longer have any fields but will simply wrap the StarlarkInfo instance and
   * delegate in each of its public methods.
   */
  @Override
  public final boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof JavaCompilationInfoProvider)) {
      return false;
    }
    JavaCompilationInfoProvider other = (JavaCompilationInfoProvider) obj;
    return Objects.equals(getJavacOpts(), other.getJavacOpts())
        && Objects.equals(getRuntimeClasspath(), other.getRuntimeClasspath())
        && Objects.equals(getCompilationClasspath(), other.getCompilationClasspath())
        && bootClasspath().shallowEquals(other.bootClasspath());
  }

  /* See comment for #equals above on why we need this. */
  @Override
  public final int hashCode() {
    return Objects.hash(
        getJavacOpts(),
        getRuntimeClasspath(),
        getCompilationClasspath(),
        bootClasspath().shallowHashCode());
  }
}
