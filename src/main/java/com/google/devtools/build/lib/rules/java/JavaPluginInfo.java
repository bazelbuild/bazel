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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo.JavaPluginData;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaPluginInfoApi;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.syntax.Location;

/** Provider for users of Java plugins. */
@Immutable
@AutoValue
public abstract class JavaPluginInfo extends NativeInfo
    implements JavaPluginInfoApi<Artifact, JavaPluginData, JavaOutput> {
  public static final String PROVIDER_NAME = "JavaPluginInfo";
  public static final Provider PROVIDER = new Provider();

  private static final JavaPluginInfo EMPTY =
      new AutoValue_JavaPluginInfo(
          ImmutableList.of(), JavaPluginData.empty(), JavaPluginData.empty());

  public static ImmutableList<JavaPluginInfo> wrapSequence(Sequence<?> sequence, String what)
      throws EvalException {
    ImmutableList.Builder<JavaPluginInfo> builder = ImmutableList.builder();
    Sequence<Info> plugins = Sequence.cast(sequence, Info.class, what);
    for (int i = 0; i < plugins.size(); i++) {
      try {
        builder.add(PROVIDER.wrap(plugins.get(i)));
      } catch (RuleErrorException e) {
        throw Starlark.errorf("at index %s of %s, %s", i, what, e.getMessage());
      }
    }
    return builder.build();
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /** Provider class for {@link JavaPluginInfo} objects. */
  public static class Provider extends StarlarkProviderWrapper<JavaPluginInfo>
      implements com.google.devtools.build.lib.packages.Provider {
    private Provider() {
      super(Label.parseCanonicalUnchecked("@_builtins//:common/java/java_info.bzl"), PROVIDER_NAME);
    }

    @Override
    public boolean isExported() {
      return true;
    }

    @Override
    public String getPrintableName() {
      return PROVIDER_NAME;
    }

    @Override
    public Location getLocation() {
      return Location.BUILTIN;
    }

    @Override
    public JavaPluginInfo wrap(Info value) throws RuleErrorException {
      if (value instanceof JavaInfo) {
        // needed because currently native JavaInfo extends JavaPluginInfo
        throw new RuleErrorException("got element of type JavaInfo, want JavaPluginInfo");
      } else if (value instanceof JavaPluginInfo) {
        return (JavaPluginInfo) value;
      } else if (value instanceof StructImpl) {
        try {
          StructImpl info = (StructImpl) value;
          return new AutoValue_JavaPluginInfo(
              JavaOutput.wrapSequence(
                  Sequence.cast(info.getValue("java_outputs"), Object.class, "java_outputs")),
              JavaPluginData.wrap(info.getValue("plugins")),
              JavaPluginData.wrap(info.getValue("api_generating_plugins")));
        } catch (EvalException e) {
          throw new RuleErrorException(e);
        }
      } else {
        throw new RuleErrorException(
            "got element of type " + Starlark.type(value) + ", want JavaPluginInfo");
      }
    }
  }

  /** Information about a Java plugin, except for whether it generates API. */
  @Immutable
  @AutoValue
  public abstract static class JavaPluginData implements JavaPluginInfoApi.JavaPluginDataApi {
    private static final JavaPluginData EMPTY =
        new AutoValue_JavaPluginInfo_JavaPluginData(
            NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
            NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
            NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER));

    public static JavaPluginData create(
        NestedSet<String> processorClasses,
        NestedSet<Artifact> processorClasspath,
        NestedSet<Artifact> data) {
      if (processorClasses.isEmpty() && processorClasspath.isEmpty() && data.isEmpty()) {
        return empty();
      }
      return new AutoValue_JavaPluginInfo_JavaPluginData(
          processorClasses, processorClasspath, data);
    }

    public static JavaPluginData empty() {
      return EMPTY;
    }

    public static JavaPluginData merge(Iterable<JavaPluginData> plugins) {
      NestedSetBuilder<String> processorClasses = NestedSetBuilder.naiveLinkOrder();
      NestedSetBuilder<Artifact> processorClasspath = NestedSetBuilder.naiveLinkOrder();
      NestedSetBuilder<Artifact> data = NestedSetBuilder.naiveLinkOrder();
      for (JavaPluginData plugin : plugins) {
        processorClasses.addTransitive(plugin.processorClasses());
        processorClasspath.addTransitive(plugin.processorClasspath());
        data.addTransitive(plugin.data());
      }
      return create(processorClasses.build(), processorClasspath.build(), data.build());
    }

    public static JavaPluginData wrap(Object obj) throws EvalException, RuleErrorException {
      if (obj instanceof JavaPluginData) {
        return (JavaPluginData) obj;
      } else if (obj instanceof StructImpl) {
        StructImpl struct = (StructImpl) obj;
        return JavaPluginData.create(
            Depset.cast(struct.getValue("processor_classes"), String.class, "processor_classes"),
            Depset.cast(struct.getValue("processor_jars"), Artifact.class, "processor_jars"),
            Depset.cast(struct.getValue("processor_data"), Artifact.class, "processor_data"));
      }
      throw new RuleErrorException("Should never happen! Got unexpected type: " + obj.getClass());
    }

    /**
     * Returns the class that should be passed to javac in order to run the annotation processor
     * this class represents.
     */
    public abstract NestedSet<String> processorClasses();

    /** Returns the artifacts to add to the runtime classpath for this plugin. */
    public abstract NestedSet<Artifact> processorClasspath();

    public abstract NestedSet<Artifact> data();

    @Override
    public Depset /*<FileApi>*/ getProcessorJarsForStarlark() {
      return Depset.of(Artifact.class, processorClasspath());
    }

    @Override
    public Depset /*<String>*/ getProcessorClassesForStarlark() {
      return Depset.of(String.class, processorClasses());
    }

    @Override
    public Depset /*<FileApi>*/ getProcessorDataForStarlark() {
      return Depset.of(Artifact.class, data());
    }

    public boolean isEmpty() {
      return processorClasses().isEmpty() && processorClasspath().isEmpty() && data().isEmpty();
    }

    private JavaPluginData disableAnnotationProcessing() {
      return JavaPluginData.create(
          /* processorClasses= */ NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
          // Preserve the processor path, since it may contain Error Prone plugins which
          // will be service-loaded by JavaBuilder.
          processorClasspath(),
          // Preserve data, which may be used by Error Prone plugins.
          data());
    }

    @Override
    public String toProto() throws EvalException {
      throw Starlark.errorf("unsupported method");
    }

    @Override
    public String toJson() throws EvalException {
      throw Starlark.errorf("unsupported method");
    }
  }

  public static JavaPluginInfo mergeWithoutJavaOutputs(JavaPluginInfo a, JavaPluginInfo b) {
    return a.isEmpty() ? b : b.isEmpty() ? a : mergeWithoutJavaOutputs(ImmutableList.of(a, b));
  }

  public static JavaPluginInfo mergeWithoutJavaOutputs(Iterable<JavaPluginInfo> providers) {
    List<JavaPluginData> plugins = new ArrayList<>();
    List<JavaPluginData> apiGeneratingPlugins = new ArrayList<>();
    for (JavaPluginInfo provider : providers) {
      if (!provider.plugins().isEmpty()) {
        plugins.add(provider.plugins());
      }
      if (!provider.apiGeneratingPlugins().isEmpty()) {
        apiGeneratingPlugins.add(provider.apiGeneratingPlugins());
      }
    }
    if (plugins.isEmpty() && apiGeneratingPlugins.isEmpty()) {
      return JavaPluginInfo.empty();
    }
    return new AutoValue_JavaPluginInfo(
        ImmutableList.of(),
        JavaPluginData.merge(plugins),
        JavaPluginData.merge(apiGeneratingPlugins));
  }

  public static JavaPluginInfo create(
      JavaPluginData javaPluginData, boolean generatesApi, ImmutableList<JavaOutput> javaOutputs) {
    return new AutoValue_JavaPluginInfo(
        javaOutputs, javaPluginData, generatesApi ? javaPluginData : JavaPluginData.empty());
  }

  public static JavaPluginInfo empty() {
    return EMPTY;
  }

  @Override
  public abstract JavaPluginData plugins();

  @Override
  public abstract JavaPluginData apiGeneratingPlugins();

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

  /**
   * Returns a copy of this {@code JavaPluginInfo} with annotation processors disabled. Does not
   * remove the processor path or data, which may be needed for Error Prone plugins.
   */
  public JavaPluginInfo disableAnnotationProcessing() {
    return JavaPluginInfo.create(
        plugins().disableAnnotationProcessing(), /* generatesApi= */ false, getJavaOutputs());
  }

  /**
   * Translates the plugin information from a {@link JavaInfo} instance.
   *
   * @param javaInfo the {@link JavaInfo} instance
   * @return a {@link JavaPluginInfo} instance
   * @throws EvalException if there are any errors accessing Starlark values
   * @throws RuleErrorException if the {@code plugins} or {@code api_generating_plugins} fields are
   *     of an incompatible type
   */
  static JavaPluginInfo fromStarlarkJavaInfo(StructImpl javaInfo)
      throws EvalException, RuleErrorException {
    JavaPluginData plugins = JavaPluginData.wrap(javaInfo.getValue("plugins"));
    JavaPluginData apiGeneratingPlugins =
        JavaPluginData.wrap(javaInfo.getValue("api_generating_plugins"));
    if (plugins.isEmpty() && apiGeneratingPlugins.isEmpty()) {
      return JavaPluginInfo.empty();
    }
    return new AutoValue_JavaPluginInfo(ImmutableList.of(), plugins, apiGeneratingPlugins);
  }
}
