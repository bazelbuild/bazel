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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.ArrayList;
import java.util.List;

/** Provider for users of Java plugins. */
@AutoCodec
@Immutable
@AutoValue
public abstract class JavaPluginInfoProvider implements TransitiveInfoProvider {

  /** Information about a Java plugin, except for whether it generates API. */
  @AutoCodec
  @Immutable
  @AutoValue
  public abstract static class JavaPluginInfo {

    public static JavaPluginInfo create(
        NestedSet<String> processorClasses,
        NestedSet<Artifact> processorClasspath,
        NestedSet<Artifact> data) {
      return new AutoValue_JavaPluginInfoProvider_JavaPluginInfo(
          processorClasses, processorClasspath, data);
    }

    @AutoCodec.Instantiator
    public static JavaPluginInfo empty() {
      return create(
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER));
    }

    public static JavaPluginInfo merge(Iterable<JavaPluginInfo> plugins) {
      NestedSetBuilder<String> processorClasses = NestedSetBuilder.naiveLinkOrder();
      NestedSetBuilder<Artifact> processorClasspath = NestedSetBuilder.naiveLinkOrder();
      NestedSetBuilder<Artifact> data = NestedSetBuilder.naiveLinkOrder();
      for (JavaPluginInfo plugin : plugins) {
        processorClasses.addTransitive(plugin.processorClasses());
        processorClasspath.addTransitive(plugin.processorClasspath());
        data.addTransitive(plugin.data());
      }
      return create(processorClasses.build(), processorClasspath.build(), data.build());
    }

    /**
     * Returns the class that should be passed to javac in order to run the annotation processor
     * this class represents.
     */
    public abstract NestedSet<String> processorClasses();

    /** Returns the artifacts to add to the runtime classpath for this plugin. */
    public abstract NestedSet<Artifact> processorClasspath();

    public abstract NestedSet<Artifact> data();

    public boolean isEmpty() {
      return processorClasses().isEmpty() && processorClasspath().isEmpty() && data().isEmpty();
    }
  }

  public static JavaPluginInfoProvider merge(JavaPluginInfoProvider a, JavaPluginInfoProvider b) {
    return a.isEmpty() ? b : b.isEmpty() ? a : merge(ImmutableList.of(a, b));
  }

  public static JavaPluginInfoProvider merge(Iterable<JavaPluginInfoProvider> providers) {
    List<JavaPluginInfo> plugins = new ArrayList<>();
    List<JavaPluginInfo> apiGeneratingPlugins = new ArrayList<>();
    for (JavaPluginInfoProvider provider : providers) {
      plugins.add(provider.plugins());
      apiGeneratingPlugins.add(provider.apiGeneratingPlugins());
    }
    return new AutoValue_JavaPluginInfoProvider(
        JavaPluginInfo.merge(plugins), JavaPluginInfo.merge(apiGeneratingPlugins));
  }

  public static JavaPluginInfoProvider create(JavaPluginInfo javaPluginInfo, Boolean generatesApi) {
    return new AutoValue_JavaPluginInfoProvider(
        javaPluginInfo, generatesApi ? javaPluginInfo : JavaPluginInfo.empty());
  }

  @AutoCodec.Instantiator
  public static JavaPluginInfoProvider empty() {
    return new AutoValue_JavaPluginInfoProvider(JavaPluginInfo.empty(), JavaPluginInfo.empty());
  }

  public abstract JavaPluginInfo plugins();

  public abstract JavaPluginInfo apiGeneratingPlugins();

  /** Returns true if the provider has no associated data. */
  public boolean isEmpty() {
    // apiGeneratingPlugins is a subset of plugins, so checking if plugins is empty is sufficient
    return plugins().isEmpty();
  }

  /**
   * Returns true if the provider has any associated annotation processors (regardless of whether it
   * has a classpath or data).
   */
  public boolean hasProcessors() {
    // apiGeneratingPlugins is a subset of plugins, so checking if plugins is empty is sufficient
    return !plugins().processorClasses().isEmpty();
  }
}
