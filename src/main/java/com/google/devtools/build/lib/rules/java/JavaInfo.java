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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.Depset.TypeException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkProviderWrapper;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo.JavaPluginData;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaModuleFlagsProviderApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.Location;

/** A Starlark declared provider that encapsulates all providers that are needed by Java rules. */
@Immutable
public final class JavaInfo extends NativeInfo
    implements JavaInfoApi<Artifact, JavaOutput, JavaPluginData> {

  public static final String STARLARK_NAME = "JavaInfo";

  public static final JavaInfoProvider PROVIDER = new JavaInfoProvider();

  public static boolean isJavaTarget(TransitiveInfoCollection target) throws RuleErrorException {
    return JavaInfo.getCompilationArgsProvider(target).isPresent();
  }

  public static NestedSet<Artifact> transitiveRuntimeJars(TransitiveInfoCollection target)
      throws RuleErrorException {
    return transformStarlarkDepsetApi(target, JavaInfo::getTransitiveRuntimeJars);
  }

  public static NestedSet<Artifact> transitiveCompileTimeJars(TransitiveInfoCollection target)
      throws RuleErrorException {
    return transformStarlarkDepsetApi(target, JavaInfo::getTransitiveCompileTimeJars);
  }

  private static NestedSet<Artifact> transformStarlarkDepsetApi(
      TransitiveInfoCollection target, Function<JavaInfo, Depset> api) throws RuleErrorException {
    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    if (javaInfo != null) {
      try {
        return api.apply(javaInfo).getSet(Artifact.class);
      } catch (TypeException e) {
        throw new RuleErrorException(e.getMessage());
      }
    }
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  public static NestedSet<Artifact> bootClasspath(TransitiveInfoCollection target)
      throws RuleErrorException {
    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    if (javaInfo != null && javaInfo.providerJavaCompilationInfo != null) {
      return javaInfo.providerJavaCompilationInfo.bootClasspath();
    }
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  public static Optional<Artifact> genSourceJar(TransitiveInfoCollection target)
      throws RuleErrorException, EvalException {
    Optional<JavaGenJarsProvider> genJarsProviderOpt =
        Optional.ofNullable(getJavaInfo(target)).map(javaInfo -> javaInfo.providerJavaGenJars);
    if (genJarsProviderOpt.isPresent()) {
      return Optional.ofNullable(genJarsProviderOpt.get().getGenSourceJar());
    } else {
      return Optional.empty();
    }
  }

  public static Optional<JavaCompilationArgsProvider> getCompilationArgsProvider(
      TransitiveInfoCollection target) throws RuleErrorException {
    return Optional.ofNullable(getJavaInfo(target))
        .map(javaInfo -> javaInfo.providerJavaCompilationArgs);
  }

  public static NestedSet<Artifact> transitiveFullCompileTimeJars(TransitiveInfoCollection target)
      throws RuleErrorException {
    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    if (javaInfo != null && javaInfo.providerJavaCompilationArgs != null) {
      return javaInfo.providerJavaCompilationArgs.getTransitiveFullCompileTimeJars();
    }
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  public static CcInfo ccInfo(TransitiveInfoCollection target) throws RuleErrorException {
    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    if (javaInfo != null && javaInfo.providerJavaCcInfo != null) {
      return javaInfo.providerJavaCcInfo.getCcInfo();
    }
    return CcInfo.EMPTY;
  }

  public static NestedSet<Artifact> transitiveSourceJars(TransitiveInfoCollection target)
      throws RuleErrorException {
    return transformStarlarkDepsetApi(target, JavaInfo::getTransitiveSourceJars);
  }

  public static JavaGenJarsProvider genJarsProvider(TransitiveInfoCollection target)
      throws RuleErrorException {
    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    if (javaInfo != null && javaInfo.providerJavaGenJars != null) {
      return javaInfo.providerJavaGenJars;
    }
    return JavaGenJarsProvider.EMPTY;
  }

  public static JavaModuleFlagsProvider moduleFlagsProvider(TransitiveInfoCollection target)
      throws RuleErrorException {
    JavaInfo javaInfo = JavaInfo.getJavaInfo(target);
    if (javaInfo != null && javaInfo.providerModuleFlags != null) {
      return javaInfo.providerModuleFlags;
    }
    return JavaModuleFlagsProvider.EMPTY;
  }

  public static ImmutableList<NestedSet<LibraryToLink>> transitiveCcNativeLibraries(
      Collection<? extends TransitiveInfoCollection> targets) throws RuleErrorException {
    ImmutableList.Builder<NestedSet<LibraryToLink>> builder = ImmutableList.builder();
    for (TransitiveInfoCollection target : targets) {
      CcInfo ccInfo = ccInfo(target);
      builder.add(ccInfo.getCcNativeLibraryInfo().getTransitiveCcNativeLibraries());
    }
    return builder.build();
  }

  public static ImmutableList<CcInfo> ccInfos(Iterable<? extends TransitiveInfoCollection> targets)
      throws RuleErrorException {
    ImmutableList.Builder<CcInfo> builder = ImmutableList.builder();
    for (TransitiveInfoCollection target : targets) {
      builder.add(JavaInfo.ccInfo(target));
    }
    return builder.build();
  }

  public static ImmutableList<JavaInfo> wrapSequence(Sequence<?> sequence, String what)
      throws EvalException {
    ImmutableList.Builder<JavaInfo> builder = ImmutableList.builder();
    Sequence<Info> infos = Sequence.cast(sequence, Info.class, what);
    for (int i = 0; i < infos.size(); i++) {
      try {
        builder.add(PROVIDER.wrap(infos.get(i)));
      } catch (RuleErrorException e) {
        throw Starlark.errorf("at index %s of %s, %s", i, what, e.getMessage());
      }
    }
    return builder.build();
  }

  public static boolean isJavaInfo(Object obj) {
    return JavaStarlarkCommon.isInstanceOfProvider(obj, JavaInfo.PROVIDER);
  }

  public Optional<JavaCompilationArgsProvider> compilationArgsProvider() {
    return Optional.ofNullable(providerJavaCompilationArgs);
  }

  /** Marker interface for encapuslated providers */
  public interface JavaInfoInternalProvider {}

  @Nullable
  static <T> T nullIfNone(Object object, Class<T> type) {
    return object != Starlark.NONE ? type.cast(object) : null;
  }

  public static final JavaInfo EMPTY = JavaInfo.Builder.create().build();

  private final JavaCompilationArgsProvider providerJavaCompilationArgs;
  private final JavaSourceJarsProvider providerJavaSourceJars;
  private final JavaRuleOutputJarsProvider providerJavaRuleOutputJars;
  private final JavaGenJarsProvider providerJavaGenJars;
  private final JavaCompilationInfoProvider providerJavaCompilationInfo;
  private final JavaCcInfoProvider providerJavaCcInfo;
  private final JavaModuleFlagsProvider providerModuleFlags;
  private final JavaPluginInfo providerJavaPlugin;

  /*
   * Contains the .jar files to be put on the runtime classpath by the configured target.
   * <p>Unlike {@link JavaCompilationArgs#getRuntimeJars()}, it does not contain transitive runtime
   * jars, only those produced by the configured target itself.
   *
   * <p>The reason why this field exists is that neverlink libraries do not contain the compiled jar
   * in {@link JavaCompilationArgs#getRuntimeJars()} and those are sometimes needed, for example,
   * for Proguarding (the compile time classpath is not enough because that contains only ijars)
   */
  private final ImmutableList<Artifact> directRuntimeJars;

  /** Java constraints (e.g. "android") that are present on the target. */
  private final ImmutableList<String> javaConstraints;

  // Whether this library should be used only for compilation and not at runtime.
  private final boolean neverlink;

  @Nullable
  public JavaPluginInfo getJavaPluginInfo() {
    return providerJavaPlugin;
  }

  /**
   * Returns a stream of providers of the specified class, fetched from the given list of {@link
   * JavaInfo}.
   */
  public static <C extends JavaInfoInternalProvider> Stream<C> streamProviders(
      Iterable<JavaInfo> javaProviders, Class<C> providerClass) {
    return Streams.stream(javaProviders)
        .map(javaInfo -> javaInfo.getProvider(providerClass))
        .filter(Objects::nonNull);
  }

  /** Returns the instance for the provided providerClass, or <tt>null</tt> if not present. */
  // TODO(adonovan): rename these three overloads of getProvider to avoid
  // confusion with the unrelated no-arg Info.getProvider method.
  @SuppressWarnings({"UngroupedOverloads", "unchecked"})
  @Nullable
  public <P extends JavaInfoInternalProvider> P getProvider(Class<P> providerClass) {
    if (providerClass == JavaCompilationArgsProvider.class) {
      return (P) providerJavaCompilationArgs;
    } else if (providerClass == JavaSourceJarsProvider.class) {
      return (P) providerJavaSourceJars;
    } else if (providerClass == JavaRuleOutputJarsProvider.class) {
      return (P) providerJavaRuleOutputJars;
    } else if (providerClass == JavaGenJarsProvider.class) {
      return (P) providerJavaGenJars;
    } else if (providerClass == JavaCompilationInfoProvider.class) {
      return (P) providerJavaCompilationInfo;
    } else if (providerClass == JavaCcInfoProvider.class) {
      return (P) providerJavaCcInfo;
    } else if (providerClass == JavaModuleFlagsProvider.class) {
      return (P) providerModuleFlags;
    }
    throw new IllegalArgumentException("unexpected provider: " + providerClass);
  }

  /** Returns a provider of the specified class, fetched from the JavaInfo of the given target. */
  @Nullable
  @VisibleForTesting
  public static <T extends JavaInfoInternalProvider> T getProvider(
      Class<T> providerClass, TransitiveInfoCollection target) throws RuleErrorException {
    JavaInfo javaInfo = getJavaInfo(target);
    if (javaInfo == null) {
      return null;
    }
    return javaInfo.getProvider(providerClass);
  }

  public static JavaInfo getJavaInfo(TransitiveInfoCollection target) throws RuleErrorException {
    return target.get(JavaInfo.PROVIDER);
  }

  private JavaInfo(
      JavaCcInfoProvider javaCcInfoProvider,
      JavaCompilationArgsProvider javaCompilationArgsProvider,
      JavaCompilationInfoProvider javaCompilationInfoProvider,
      JavaGenJarsProvider javaGenJarsProvider,
      JavaModuleFlagsProvider javaModuleFlagsProvider,
      JavaPluginInfo javaPluginInfo,
      JavaRuleOutputJarsProvider javaRuleOutputJarsProvider,
      JavaSourceJarsProvider javaSourceJarsProvider,
      ImmutableList<Artifact> directRuntimeJars,
      boolean neverlink,
      ImmutableList<String> javaConstraints,
      Location creationLocation) {
    super(creationLocation);
    this.directRuntimeJars = directRuntimeJars;
    this.neverlink = neverlink;
    this.javaConstraints = javaConstraints;
    this.providerJavaCcInfo = javaCcInfoProvider;
    this.providerJavaCompilationArgs = javaCompilationArgsProvider;
    this.providerJavaCompilationInfo = javaCompilationInfoProvider;
    this.providerJavaGenJars = javaGenJarsProvider;
    this.providerModuleFlags = javaModuleFlagsProvider;
    this.providerJavaPlugin = javaPluginInfo;
    this.providerJavaRuleOutputJars = javaRuleOutputJarsProvider;
    this.providerJavaSourceJars = javaSourceJarsProvider;
  }

  private JavaInfo(StructImpl javaInfo) throws EvalException, TypeException, RuleErrorException {
    this(
        JavaCcInfoProvider.fromStarlarkJavaInfo(javaInfo),
        JavaCompilationArgsProvider.fromStarlarkJavaInfo(javaInfo),
        JavaCompilationInfoProvider.fromStarlarkJavaInfo(javaInfo),
        JavaGenJarsProvider.from(javaInfo.getValue("annotation_processing")),
        JavaModuleFlagsProvider.fromStarlarkJavaInfo(javaInfo),
        JavaPluginInfo.fromStarlarkJavaInfo(javaInfo),
        JavaRuleOutputJarsProvider.fromStarlarkJavaInfo(javaInfo),
        JavaSourceJarsProvider.fromStarlarkJavaInfo(javaInfo),
        extractDirectRuntimeJars(javaInfo),
        extractNeverLink(javaInfo),
        extractConstraints(javaInfo),
        javaInfo.getCreationLocation());
  }

  private static ImmutableList<Artifact> extractDirectRuntimeJars(StructImpl javaInfo)
      throws EvalException {
    return Sequence.cast(
            javaInfo.getValue("runtime_output_jars"), Artifact.class, "runtime_output_jars")
        .getImmutableList();
  }

  private static boolean extractNeverLink(StructImpl javaInfo) throws EvalException {
    Boolean neverlink = nullIfNone(javaInfo.getValue("_neverlink"), Boolean.class);
    return neverlink != null && neverlink;
  }

  private static ImmutableList<String> extractConstraints(StructImpl javaInfo)
      throws EvalException {
    Object constraints = javaInfo.getValue("_constraints");
    if (constraints == null || constraints == Starlark.NONE) {
      return ImmutableList.of();
    }
    return Sequence.cast(constraints, String.class, "_constraints").getImmutableList();
  }

  @Override
  public JavaInfoProvider getProvider() {
    return PROVIDER;
  }

  @Override
  public boolean isNeverlink() {
    return neverlink;
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveRuntimeJars() {
    return Depset.of(
        Artifact.class,
        getProviderAsNestedSet(
            JavaCompilationArgsProvider.class, JavaCompilationArgsProvider::getRuntimeJars));
  }

  @Override
  public Depset /*<Artifact>*/ getTransitiveCompileTimeJars() {
    return Depset.of(
        Artifact.class,
        getProviderAsNestedSet(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider::getTransitiveCompileTimeJars));
  }

  @Override
  public Depset /*<Artifact>*/ getCompileTimeJars() {
    NestedSet<Artifact> compileTimeJars =
        getProviderAsNestedSet(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider::getDirectCompileTimeJars);
    return Depset.of(Artifact.class, compileTimeJars);
  }

  @Override
  public Depset getFullCompileTimeJars() {
    NestedSet<Artifact> fullCompileTimeJars =
        getProviderAsNestedSet(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider::getDirectFullCompileTimeJars);
    return Depset.of(Artifact.class, fullCompileTimeJars);
  }

  @Override
  public Sequence<Artifact> getSourceJars() {
    // TODO(#4221) change return type to NestedSet<Artifact>
    ImmutableList<Artifact> sourceJars =
        providerJavaSourceJars == null
            ? ImmutableList.of()
            : providerJavaSourceJars.getSourceJars();
    return StarlarkList.immutableCopyOf(sourceJars);
  }

  @Override
  @Deprecated
  public JavaRuleOutputJarsProvider getOutputJars() {
    return providerJavaRuleOutputJars;
  }

  @Override
  public ImmutableList<JavaOutput> getJavaOutputs() {
    return providerJavaRuleOutputJars == null
        ? ImmutableList.of()
        : providerJavaRuleOutputJars.getJavaOutputs();
  }

  @Override
  public JavaGenJarsProvider getGenJarsProvider() {
    return providerJavaGenJars;
  }

  @Override
  public JavaCompilationInfoProvider getCompilationInfoProvider() {
    return providerJavaCompilationInfo;
  }

  @Override
  public Sequence<Artifact> getRuntimeOutputJars() {
    return StarlarkList.immutableCopyOf(getDirectRuntimeJars());
  }

  public ImmutableList<Artifact> getDirectRuntimeJars() {
    return directRuntimeJars;
  }

  /*<Artifact>*/

  /*<Artifact>*/

  @Override
  public Depset /*<Artifact>*/ getTransitiveSourceJars() {
    return Depset.of(
        Artifact.class,
        getProviderAsNestedSet(
            JavaSourceJarsProvider.class, JavaSourceJarsProvider::getTransitiveSourceJars));
  }

  /** Returns the transitive set of CC native libraries required by the target. */
  public NestedSet<LibraryToLink> getTransitiveNativeLibraries() {
    return getProviderAsNestedSet(
        JavaCcInfoProvider.class,
        x -> x.getCcInfo().getCcNativeLibraryInfo().getTransitiveCcNativeLibraries());
  }

  @Override
  public Depset /*<LibraryToLink>*/ getTransitiveNativeLibrariesForStarlark() {
    return Depset.of(LibraryToLink.class, getTransitiveNativeLibraries());
  }

  @Override
  public CcInfoApi<Artifact> getCcLinkParamInfo() {
    return providerJavaCcInfo != null ? providerJavaCcInfo.getCcInfo() : CcInfo.EMPTY;
  }

  @Override
  public JavaModuleFlagsProviderApi getJavaModuleFlagsInfo() {
    return providerModuleFlags == null ? JavaModuleFlagsProvider.EMPTY : providerModuleFlags;
  }

  @Override
  public Depset getTransitiveFullCompileJars() {
    return Depset.of(
        Artifact.class,
        getProviderAsNestedSet(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider::getTransitiveFullCompileTimeJars));
  }

  @Override
  public Depset getCompileTimeJavaDependencies() {
    return Depset.of(
        Artifact.class,
        getProviderAsNestedSet(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider::getCompileTimeJavaDependencyArtifacts));
  }

  @Override
  public JavaPluginData plugins() {
    return providerJavaPlugin == null ? JavaPluginData.empty() : providerJavaPlugin.plugins();
  }

  @Override
  public JavaPluginData apiGeneratingPlugins() {
    return providerJavaPlugin == null
        ? JavaPluginData.empty()
        : providerJavaPlugin.apiGeneratingPlugins();
  }

  /** Returns all constraints set on the associated target. */
  public ImmutableList<String> getJavaConstraints() {
    return javaConstraints;
  }

  @Override
  public Sequence<String> getJavaConstraintsStarlark() {
    return StarlarkList.immutableCopyOf(javaConstraints);
  }

  /**
   * Gets Provider, check it for not null and call function to get NestedSet&lt;S&gt; from it.
   *
   * <p>Gets provider from map. If Provider is null, return default, empty, stabled ordered
   * NestedSet. If provider is not null, then delegates to mapper all responsibility to fetch
   * required NestedSet from provider.
   *
   * @see JavaInfo#getProviderAsNestedSet(Class, Function, Function)
   * @param providerClass provider class. used as key to look up for provider.
   * @param mapper Function used to convert provider to NesteSet&lt;S&gt;
   * @param <P> type of Provider
   * @param <S> type of returned NestedSet items
   */
  private <P extends JavaInfoInternalProvider, S extends StarlarkValue>
      NestedSet<S> getProviderAsNestedSet(
          Class<P> providerClass, Function<P, NestedSet<S>> mapper) {

    P provider = getProvider(providerClass);
    if (provider == null) {
      return NestedSetBuilder.<S>stableOrder().build();
    }
    return mapper.apply(provider);
  }

  @Override
  public boolean equals(Object otherObject) {
    if (this == otherObject) {
      return true;
    }
    if (!(otherObject instanceof JavaInfo)) {
      return false;
    }

    JavaInfo other = (JavaInfo) otherObject;
    return Objects.equals(providerJavaCompilationArgs, other.providerJavaCompilationArgs)
        && Objects.equals(providerJavaSourceJars, other.providerJavaSourceJars)
        && Objects.equals(providerJavaRuleOutputJars, other.providerJavaRuleOutputJars)
        && Objects.equals(providerJavaGenJars, other.providerJavaGenJars)
        && Objects.equals(providerJavaCompilationInfo, other.providerJavaCompilationInfo)
        && Objects.equals(providerJavaCcInfo, other.providerJavaCcInfo)
        && Objects.equals(providerModuleFlags, other.providerModuleFlags)
        && Objects.equals(providerJavaPlugin, other.providerJavaPlugin);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        providerJavaCompilationArgs,
        providerJavaSourceJars,
        providerJavaRuleOutputJars,
        providerJavaGenJars,
        providerJavaCompilationInfo,
        providerJavaCcInfo,
        providerModuleFlags,
        providerJavaPlugin);
  }

  /** Provider class for {@link JavaInfo} objects. */
  public static class JavaInfoProvider extends StarlarkProviderWrapper<JavaInfo>
      implements com.google.devtools.build.lib.packages.Provider {
    private JavaInfoProvider() {
      super(Label.parseCanonicalUnchecked("@_builtins//:common/java/java_info.bzl"), STARLARK_NAME);
    }

    @Override
    public JavaInfo wrap(Info info) throws RuleErrorException {
      if (info instanceof JavaInfo) {
        return (JavaInfo) info;
      } else if (info instanceof StructImpl) {
        try {
          return new JavaInfo((StructImpl) info);
        } catch (EvalException | TypeException e) {
          throw new RuleErrorException(e);
        }
      }
      throw new RuleErrorException("got " + Starlark.type(info) + ", wanted JavaInfo");
    }

    @Override
    public boolean isExported() {
      return true;
    }

    @Override
    public String getPrintableName() {
      return STARLARK_NAME;
    }

    @Override
    public Location getLocation() {
      return Location.BUILTIN;
    }
  }

  /** A Builder for {@link JavaInfo}. */
  public static class Builder {

    private JavaCcInfoProvider providerJavaCcInfo;
    private JavaCompilationArgsProvider providerJavaCompilationArgs;
    private JavaCompilationInfoProvider providerJavaCompilationInfo;
    private JavaGenJarsProvider providerJavaGenJars;
    private JavaModuleFlagsProvider providerModuleFlags;
    private JavaPluginInfo providerJavaPlugin;
    private JavaRuleOutputJarsProvider providerJavaRuleOutputJars;
    private JavaSourceJarsProvider providerJavaSourceJars;
    private ImmutableList<Artifact> runtimeJars;
    private ImmutableList<String> javaConstraints;
    private boolean neverlink;
    private Location creationLocation = Location.BUILTIN;

    private Builder() {}

    public static Builder create() {
      return new Builder()
          .setRuntimeJars(ImmutableList.of())
          .setJavaConstraints(ImmutableList.of());
    }

    @CanIgnoreReturnValue
    public Builder setRuntimeJars(ImmutableList<Artifact> runtimeJars) {
      this.runtimeJars = runtimeJars;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setNeverlink(boolean neverlink) {
      this.neverlink = neverlink;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setJavaConstraints(ImmutableList<String> javaConstraints) {
      this.javaConstraints = javaConstraints;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setLocation(Location location) {
      this.creationLocation = location;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder javaCcInfo(JavaCcInfoProvider provider) {
      this.providerJavaCcInfo = provider;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder javaCompilationArgs(JavaCompilationArgsProvider provider) {
      this.providerJavaCompilationArgs = provider;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder javaCompilationInfo(JavaCompilationInfoProvider provider) {
      this.providerJavaCompilationInfo = provider;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder javaGenJars(JavaGenJarsProvider provider) {
      this.providerJavaGenJars = provider;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder javaModuleFlags(JavaModuleFlagsProvider provider) {
      this.providerModuleFlags = provider;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder javaRuleOutputs(JavaRuleOutputJarsProvider provider) {
      this.providerJavaRuleOutputJars = provider;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder javaSourceJars(JavaSourceJarsProvider provider) {
      this.providerJavaSourceJars = provider;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder javaPluginInfo(JavaPluginInfo javaPluginInfo) {
      this.providerJavaPlugin = javaPluginInfo;
      return this;
    }

    public JavaInfo build() {
      return new JavaInfo(
          providerJavaCcInfo,
          providerJavaCompilationArgs,
          providerJavaCompilationInfo,
          providerJavaGenJars,
          providerModuleFlags,
          providerJavaPlugin,
          providerJavaRuleOutputJars,
          providerJavaSourceJars,
          runtimeJars,
          neverlink,
          javaConstraints,
          creationLocation);
    }
  }
}
