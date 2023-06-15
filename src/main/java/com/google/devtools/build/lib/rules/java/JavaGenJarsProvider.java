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
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.java.JavaInfo.JavaInfoInternalProvider;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaAnnotationProcessingApi;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** The collection of gen jars from the transitive closure. */
public interface JavaGenJarsProvider
    extends JavaInfoInternalProvider, JavaAnnotationProcessingApi<Artifact> {

  JavaGenJarsProvider EMPTY =
      new AutoValue_JavaGenJarsProvider_NativeJavaGenJarsProvider(
          false,
          null,
          null,
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  static JavaGenJarsProvider create(
      boolean usesAnnotationProcessing,
      @Nullable Artifact genClassJar,
      @Nullable Artifact genSourceJar,
      JavaPluginInfo plugins,
      List<JavaGenJarsProvider> transitiveJavaGenJars) {
    if (!usesAnnotationProcessing
        && genClassJar == null
        && genSourceJar == null
        && plugins.isEmpty()
        && transitiveJavaGenJars.isEmpty()) {
      return EMPTY;
    }

    NestedSetBuilder<Artifact> classJarsBuilder = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> sourceJarsBuilder = NestedSetBuilder.stableOrder();

    if (genClassJar != null) {
      classJarsBuilder.add(genClassJar);
    }
    if (genSourceJar != null) {
      sourceJarsBuilder.add(genSourceJar);
    }
    for (JavaGenJarsProvider dep : transitiveJavaGenJars) {
      classJarsBuilder.addTransitive(dep.getTransitiveGenClassJars());
      sourceJarsBuilder.addTransitive(dep.getTransitiveGenSourceJars());
    }
    return new AutoValue_JavaGenJarsProvider_NativeJavaGenJarsProvider(
        usesAnnotationProcessing,
        genClassJar,
        genSourceJar,
        plugins.plugins().processorClasspath(),
        plugins.plugins().processorClasses(),
        classJarsBuilder.build(),
        sourceJarsBuilder.build());
  }

  /** Returns a copy with the given details, preserving transitiveXxx sets. */
  default JavaGenJarsProvider withDirectInfo(
      boolean usesAnnotationProcessing,
      @Nullable Artifact genClassJar,
      @Nullable Artifact genSourceJar,
      NestedSet<Artifact> processorClasspath,
      NestedSet<String> processorClassNames)
      throws EvalException {
    // Existing Jars would be a problem b/c we can't remove them from transitiveXxx sets
    if (this.getGenClassJar() != null && !Objects.equals(this.getGenClassJar(), genClassJar)) {
      throw Starlark.errorf("Existing genClassJar: %s", this.getGenClassJar());
    }
    if (this.getGenSourceJar() != null && !Objects.equals(this.getGenSourceJar(), genSourceJar)) {
      throw Starlark.errorf("Existing genSrcJar: %s", this.getGenSourceJar());
    }
    return new AutoValue_JavaGenJarsProvider_NativeJavaGenJarsProvider(
        usesAnnotationProcessing,
        genClassJar,
        genSourceJar,
        processorClasspath,
        processorClassNames,
        addIf(getTransitiveGenClassJars(), genClassJar),
        addIf(getTransitiveGenSourceJars(), genSourceJar));
  }

  default boolean isEmpty() {
    return !usesAnnotationProcessing()
        && getGenClassJar() == null
        && getGenSourceJar() == null
        && getTransitiveGenClassJars().isEmpty()
        && getTransitiveGenSourceJars().isEmpty();
  }

  NestedSet<Artifact> getTransitiveGenClassJars();

  NestedSet<Artifact> getTransitiveGenSourceJars();

  NestedSet<Artifact> getProcessorClasspath();

  /** Natively constructed JavaGenJarsProvider */
  @Immutable
  @AutoValue
  abstract class NativeJavaGenJarsProvider implements JavaGenJarsProvider {

    @Override
    public abstract boolean usesAnnotationProcessing();

    @Nullable
    @Override
    public abstract Artifact getGenClassJar();

    @Nullable
    @Override
    public abstract Artifact getGenSourceJar();

    @Override
    public abstract NestedSet<Artifact> getProcessorClasspath();

    public abstract NestedSet<String> getProcessorClassnames();

    @Override
    public abstract NestedSet<Artifact> getTransitiveGenClassJars();

    @Override
    public abstract NestedSet<Artifact> getTransitiveGenSourceJars();

    @Override
    public Depset /*<Artifact>*/ getTransitiveGenClassJarsForStarlark() {
      return Depset.of(Artifact.class, getTransitiveGenClassJars());
    }

    @Override
    public Depset /*<Artifact>*/ getTransitiveGenSourceJarsForStarlark() {
      return Depset.of(Artifact.class, getTransitiveGenSourceJars());
    }

    @Override
    public Depset /*<Artifact>*/ getProcessorClasspathForStarlark() {
      return Depset.of(Artifact.class, getProcessorClasspath());
    }

    @Override
    public ImmutableList<String> getProcessorClassNamesList() {
      return getProcessorClassnames().toList();
    }
  }

  private static <T> NestedSet<T> addIf(NestedSet<T> set, @Nullable T element) {
    if (element == null) {
      return set;
    }
    return NestedSetBuilder.<T>stableOrder().add(element).addTransitive(set).build();
  }

}
