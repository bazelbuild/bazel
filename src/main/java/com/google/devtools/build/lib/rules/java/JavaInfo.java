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


import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import javax.annotation.Nullable;

/** A Skylark declared provider that encapsulates all providers that are needed by Java rules. */
@SkylarkModule(
    name = "JavaInfo",
    doc = "Encapsulates all information provided by Java rules.",
    category = SkylarkModuleCategory.PROVIDER)
@Immutable
@AutoCodec
public final class JavaInfo extends NativeInfo {

  public static final String SKYLARK_NAME = "JavaInfo";

  public static final JavaInfoProvider PROVIDER = new JavaInfoProvider();

  @Nullable
  private static <T> T nullIfNone(Object object, Class<T> type) {
    return object != Runtime.NONE ? type.cast(object) : null;
  }

  @Nullable
  private static Object nullIfNone(Object object) {
    return nullIfNone(object, Object.class);
  }

  public static final JavaInfo EMPTY = JavaInfo.Builder.create().build();

  private static final ImmutableSet<Class<? extends TransitiveInfoProvider>> ALLOWED_PROVIDERS =
      ImmutableSet.of(
        JavaCompilationArgsProvider.class,
        JavaSourceJarsProvider.class,
        JavaRuleOutputJarsProvider.class,
        JavaRunfilesProvider.class,
        JavaPluginInfoProvider.class,
        JavaGenJarsProvider.class,
        JavaExportsProvider.class,
        JavaCompilationInfoProvider.class
      );

  private final TransitiveInfoProviderMap providers;

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

  /**
   * Java constraints (e.g. "android") that are present on the target.
   */
  private final ImmutableList<String> javaConstraints;

  // Whether or not this library should be used only for compilation and not at runtime.
  private final boolean neverlink;

  /** Returns the instance for the provided providerClass, or <tt>null</tt> if not present. */
  @Nullable
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    return providers.getProvider(providerClass);
  }

  public TransitiveInfoProviderMap getProviders() {
    return providers;
  }

  /**
   * Merges the given providers into one {@link JavaInfo}. All the providers with the same type
   * in the given list are merged into one provider that is added to the resulting
   * {@link JavaInfo}.
   */
  public static JavaInfo merge(List<JavaInfo> providers) {
    List<JavaCompilationArgsProvider> javaCompilationArgsProviders =
        JavaInfo.fetchProvidersFromList(providers, JavaCompilationArgsProvider.class);
    List<JavaSourceJarsProvider> javaSourceJarsProviders =
        JavaInfo.fetchProvidersFromList(providers, JavaSourceJarsProvider.class);
    List<JavaRunfilesProvider> javaRunfilesProviders =
        JavaInfo.fetchProvidersFromList(providers, JavaRunfilesProvider.class);
    List<JavaPluginInfoProvider> javaPluginInfoProviders =
        JavaInfo.fetchProvidersFromList(providers, JavaPluginInfoProvider.class);
    List<JavaExportsProvider> javaExportsProviders =
        JavaInfo.fetchProvidersFromList(providers, JavaExportsProvider.class);


    Runfiles mergedRunfiles = Runfiles.EMPTY;
    for (JavaRunfilesProvider javaRunfilesProvider : javaRunfilesProviders) {
      Runfiles runfiles = javaRunfilesProvider.getRunfiles();
      mergedRunfiles = mergedRunfiles == Runfiles.EMPTY ? runfiles : mergedRunfiles.merge(runfiles);
    }

    return JavaInfo.Builder.create()
        .addProvider(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider.merge(javaCompilationArgsProviders))
        .addProvider(
          JavaSourceJarsProvider.class, JavaSourceJarsProvider.merge(javaSourceJarsProviders))
        // When a rule merges multiple JavaProviders, its purpose is to pass on information, so
        // it doesn't have any output jars.
        .addProvider(JavaRuleOutputJarsProvider.class, JavaRuleOutputJarsProvider.builder().build())
        .addProvider(JavaRunfilesProvider.class, new JavaRunfilesProvider(mergedRunfiles))
        .addProvider(
            JavaPluginInfoProvider.class, JavaPluginInfoProvider.merge(javaPluginInfoProviders))
        .addProvider(JavaExportsProvider.class, JavaExportsProvider.merge(javaExportsProviders))
        // TODO(b/65618333): add merge function to JavaGenJarsProvider. See #3769
        .build();
  }

  /**
   * Returns a list of providers of the specified class, fetched from the given list of
   * {@link JavaInfo}s.
   * Returns an empty list if no providers can be fetched.
   * Returns a list of the same size as the given list if the requested providers are of type
   * JavaCompilationArgsProvider.
   */
  public static <C extends TransitiveInfoProvider> List<C> fetchProvidersFromList(
      Iterable<JavaInfo> javaProviders, Class<C> providersClass) {
    List<C> fetchedProviders = new ArrayList<>();
    for (JavaInfo javaInfo : javaProviders) {
      C provider = javaInfo.getProvider(providersClass);
      if (provider != null) {
        fetchedProviders.add(provider);
      }
    }
    return fetchedProviders;
  }

  /**
   * Returns a provider of the specified class, fetched from the specified target or, if not found,
   * from the JavaInfo of the given target. JavaInfo can be found as a declared provider
   * in SkylarkProviders.
   * Returns null if no such provider exists.
   *
   * <p>A target can either have both the specified provider and JavaInfo that encapsulates the
   * same information, or just one of them.</p>
   */
  @Nullable
  public static <T extends TransitiveInfoProvider> T getProvider(
      Class<T> providerClass, TransitiveInfoCollection target) {
    T provider = target.getProvider(providerClass);
    if (provider != null) {
      return provider;
    }
    JavaInfo javaInfo = (JavaInfo) target.get(JavaInfo.PROVIDER.getKey());
    if (javaInfo == null) {
      return null;
    }
    return javaInfo.getProvider(providerClass);
  }

  public static JavaInfo getJavaInfo(TransitiveInfoCollection target) {
    return (JavaInfo) target.get(JavaInfo.PROVIDER.getKey());
  }

  public static <T extends TransitiveInfoProvider> T getProvider(
      Class<T> providerClass, TransitiveInfoProviderMap providerMap) {
    T provider = providerMap.getProvider(providerClass);
    if (provider != null) {
      return provider;
    }
    JavaInfo javaInfo = (JavaInfo) providerMap.getProvider(JavaInfo.PROVIDER.getKey());
    if (javaInfo == null) {
      return null;
    }
    return javaInfo.getProvider(providerClass);
  }

  public static <T extends TransitiveInfoProvider> List<T> getProvidersFromListOfTargets(
      Class<T> providerClass, Iterable<? extends TransitiveInfoCollection> targets) {
    List<T> providersList = new ArrayList<>();
    for (TransitiveInfoCollection target : targets) {
      T provider = getProvider(providerClass, target);
      if (provider != null) {
        providersList.add(provider);
      }
    }
    return providersList;
  }

  /**
   * Returns a list of the given provider class with all the said providers retrieved from the
   * given {@link JavaInfo}s.
   */
  public static <T extends TransitiveInfoProvider> ImmutableList<T>
      getProvidersFromListOfJavaProviders(
          Class<T> providerClass, Iterable<JavaInfo> javaProviders) {
    ImmutableList.Builder<T> providersList = new ImmutableList.Builder<>();
    for (JavaInfo javaInfo : javaProviders) {
      T provider = javaInfo.getProvider(providerClass);
      if (provider != null) {
        providersList.add(provider);
      }
    }
    return providersList.build();
  }

  @VisibleForSerialization
  @AutoCodec.Instantiator
  JavaInfo(
      TransitiveInfoProviderMap providers,
      ImmutableList<Artifact> directRuntimeJars,
      boolean neverlink,
      ImmutableList<String> javaConstraints,
      Location location) {
    super(PROVIDER, location);
    this.directRuntimeJars = directRuntimeJars;
    this.providers = providers;
    this.neverlink = neverlink;
    this.javaConstraints = javaConstraints;
  }

  public Boolean isNeverlink() {
    return neverlink;
  }

  @SkylarkCallable(
      name = "transitive_runtime_jars",
      doc = "Depset of runtime jars required by this target",
      structField = true
  )
  public SkylarkNestedSet getTransitiveRuntimeJars() {
    return SkylarkNestedSet.of(Artifact.class, getTransitiveRuntimeDeps());
  }

  @SkylarkCallable(
      name = "transitive_compile_time_jars",
      doc = "Depset of compile time jars recusrively required by this target. See `compile_jars` "
          + "for more details.",
      structField = true
  )
  public SkylarkNestedSet getTransitiveCompileTimeJars() {
    return SkylarkNestedSet.of(Artifact.class, getTransitiveDeps());
  }

  @SkylarkCallable(
      name = "compile_jars",
      doc = "Returns the compile time jars required by this target directly. They can be: <ul>"
          + "<li> interface jars (ijars), if an ijar tool was used, either by calling "
          + "java_common.create_provider(use_ijar=True, ...) or by passing --use_ijars on the "
          + "command line for native Java rules and `java_common.compile`</li>"
          + "<li> normal full jars, if no ijar action was requested</li>"
          + "<li> both ijars and normal full jars, if this provider was created by merging two or "
          + "more providers created with different ijar requests </li> </ul>",
      structField = true
  )
  public SkylarkNestedSet getCompileTimeJars() {
    NestedSet<Artifact> compileTimeJars =
        getProviderAsNestedSet(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider::getDirectCompileTimeJars);
    return SkylarkNestedSet.of(Artifact.class, compileTimeJars);
  }

  @SkylarkCallable(
      name = "full_compile_jars",
      doc = "Returns the full compile time jars required by this target directly. They can be <ul>"
          + "<li> the corresponding normal full jars of the ijars returned by `compile_jars`</li>"
          + "<li> the normal full jars returned by `compile_jars`</li></ul>"
          + "Note: `compile_jars` can return a mix of ijars and normal full jars. In that case, "
          + "`full_compile_jars` returns the corresponding full jars of the ijars and the remaining"
          + "normal full jars in `compile_jars`.",
      structField = true
  )
  public SkylarkNestedSet getFullCompileTimeJars() {
    NestedSet<Artifact> fullCompileTimeJars =
        getProviderAsNestedSet(
            JavaCompilationArgsProvider.class, JavaCompilationArgsProvider::getFullCompileTimeJars);
    return SkylarkNestedSet.of(Artifact.class, fullCompileTimeJars);
  }

  @SkylarkCallable(
      name = "source_jars",
      doc = "Returns a list of jar files containing all the uncompiled source files (including "
      + "those generated by annotations) from the target itself, i.e. NOT including the sources of "
      + "the transitive dependencies",
      structField = true
  )
  public SkylarkList<Artifact> getSourceJars() {
    //TODO(#4221) change return type to NestedSet<Artifact>
    JavaSourceJarsProvider provider = providers.getProvider(JavaSourceJarsProvider.class);
    ImmutableList<Artifact> sourceJars =
        provider == null ? ImmutableList.of() : provider.getSourceJars();
    return SkylarkList.createImmutable(sourceJars);
  }

  @SkylarkCallable(
    name = "outputs",
    doc = "Returns information about outputs of this Java target.",
    structField = true,
    allowReturnNones = true
  )
  public JavaRuleOutputJarsProvider getOutputJars() {
    return getProvider(JavaRuleOutputJarsProvider.class);
  }


  @SkylarkCallable(
      name = "annotation_processing",
      structField = true,
      allowReturnNones = true,
      doc = "Returns information about annotation processing for this Java target."
  )
  public JavaGenJarsProvider getGenJarsProvider() {
    return getProvider(JavaGenJarsProvider.class);
  }

  @SkylarkCallable(
    name = "compilation_info",
    structField = true,
    allowReturnNones = true,
    doc = "Returns compilation information for this Java target."
  )
  public JavaCompilationInfoProvider getCompilationInfoProvider() {
    return getProvider(JavaCompilationInfoProvider.class);
  }

  @SkylarkCallable(
      name = "runtime_output_jars",
      doc = "Returns the runtime output jars provided by this Java target.",
      structField = true)
  public SkylarkList<Artifact> getRuntimeOutputJars() {
    return SkylarkList.createImmutable(getDirectRuntimeJars());
  }

  public ImmutableList<Artifact> getDirectRuntimeJars() {
    return directRuntimeJars;
  }

  @SkylarkCallable(
    name = "transitive_deps",
    doc = "Returns the transitive set of Jars required to build the target.",
    structField = true
  )
  public NestedSet<Artifact> getTransitiveDeps() {
    return getProviderAsNestedSet(
        JavaCompilationArgsProvider.class,
        JavaCompilationArgsProvider::getTransitiveCompileTimeJars);
  }

  @SkylarkCallable(
    name = "transitive_runtime_deps",
    doc = "Returns the transitive set of Jars required on the target's runtime classpath.",
    structField = true
  )
  public NestedSet<Artifact> getTransitiveRuntimeDeps() {
    return getProviderAsNestedSet(
        JavaCompilationArgsProvider.class, JavaCompilationArgsProvider::getRuntimeJars);
  }

  @SkylarkCallable(
    name = "transitive_source_jars",
    doc = "Returns the Jars containing Java source files for the target "
            + "and all of its transitive dependencies.",
    structField = true
  )
  public NestedSet<Artifact> getTransitiveSourceJars() {
    return getProviderAsNestedSet(
        JavaSourceJarsProvider.class,
        JavaSourceJarsProvider::getTransitiveSourceJars);
  }

  @SkylarkCallable(
      name = "transitive_exports",
      structField = true,
      doc = "Returns transitive set of labels that are being exported from this rule."
  )
  public NestedSet<Label> getTransitiveExports() {
    return getProviderAsNestedSet(
        JavaExportsProvider.class,
        JavaExportsProvider::getTransitiveExports);
  }

  /**
   * Returns all constraints set on the associated target.
   */
  public ImmutableList<String> getJavaConstraints() {
    return javaConstraints;
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
  private <P extends TransitiveInfoProvider, S extends SkylarkValue>
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
    return providers.equals(other.providers);
  }

  @Override
  public int hashCode() {
    return providers.hashCode();
  }

  /** Provider class for {@link JavaInfo} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  public static class JavaInfoProvider extends BuiltinProvider<JavaInfo> implements ProviderApi {
    private JavaInfoProvider() {
      super(SKYLARK_NAME, JavaInfo.class);
    }

    @SkylarkCallable(name = "JavaInfo",
        doc = "The <code>JavaInfo</code> constructor.",
        parameters = {
            @Param(
                name = "output_jar",
                type = FileApi.class,
                named = true,
                doc = "The jar that was created as a result of a compilation "
                    + "(e.g. javac, scalac, etc)."
            ),
            @Param(
                name = "compile_jar",
                type = FileApi.class,
                named = true,
                noneable = true,
                defaultValue = "None",
                doc = "A jar that is added as the compile-time dependency in lieu of "
                    + "<code>output_jar</code>. Typically this is the ijar produced by "
                    + "<code><a class=\"anchor\" href=\"java_common.html#run_ijar\">"
                    + "run_ijar</a></code>. "
                    + "If you cannot use ijar, consider instead using the output of "
                    + "<code><a class=\"anchor\" href=\"java_common.html#stamp_jar\">"
                    + "stamp_ijar</a></code>. If you do not wish to use either, "
                    + "you can simply pass <code>output_jar</code>."
            ),
            @Param(
                name = "source_jar",
                type = FileApi.class,
                named = true,
                noneable = true,
                defaultValue = "None",
                doc = "The source jar that was used to create the output jar. "
                    + "Use <code><a class=\"anchor\" href=\"java_common.html#pack_sources\">"
                    + "pack_sources</a></code> to produce this source jar."
            ),
            @Param(
                name = "neverlink",
                type = Boolean.class,
                named = true,
                defaultValue = "False",
                doc = "If true only use this library for compilation and not at runtime."
            ),
            @Param(
                name = "deps",
                type = SkylarkList.class,
                generic1 = JavaInfo.class,
                named = true,
                defaultValue = "[]",
                doc = "Compile time dependencies that were used to create the output jar."
            ),
            @Param(
                name = "runtime_deps",
                type = SkylarkList.class,
                generic1 = JavaInfo.class,
                named = true,
                defaultValue = "[]",
                doc = "Runtime dependencies that are needed for this library."
            ),
            @Param(
                name = "exports",
                type = SkylarkList.class,
                generic1 = JavaInfo.class,
                named = true,
                defaultValue = "[]",
                doc = "Libraries to make available for users of this library. See also "
                    + "<a class=\"anchor\" href=\"https://docs.bazel.build/versions/"
                    + "master/be/java.html#java_library.exports\">java_library.exports</a>."
            ),
            @Param(
                name = "actions",
                type = SkylarkActionFactoryApi.class,
                named = true,
                defaultValue = "None",
                noneable = true,
                doc = "Deprecated. No longer needed when <code>compile_jar</code> and/or "
                    + "<code>source_jar</code> are used. "
                    + "<p>Used to create the ijar and pack source files to jar actions.</p>"
            ),
            @Param(
                name = "sources",
                type = SkylarkList.class,
                generic1 = FileApi.class,
                named = true,
                defaultValue = "None",
                noneable = true,
                doc = "Deprecated. Use <code>source_jar</code> instead. "
                    + "<p>The sources that were used to create the output jar.</p>"
            ),
            @Param(
                name = "source_jars",
                type = SkylarkList.class,
                generic1 = FileApi.class,
                named = true,
                defaultValue = "None",
                noneable = true,
                doc = "Deprecated. Use <code>source_jar</code> instead. "
                    + "<p>The source jars that were used to create the output jar.</p>"
            ),
            @Param(
                name = "use_ijar",
                type = Boolean.class,
                named = true,
                defaultValue = "None",
                noneable = true,
                doc = "Deprecated. Use <code>compile_jar</code> instead. "
                    + "<p>If an ijar of the output jar should be created and stored in the "
                    + "provider. </p>"
            ),
            @Param(
                name = "java_toolchain",
                type = ConfiguredTarget.class,
                named = true,
                defaultValue = "None",
                noneable = true,
                doc = "Deprecated. No longer needed when <code>compile_jar</code> and/or "
                    + "<code>source_jar</code> are used. "
                    + "<p>The toolchain to be used for retrieving the ijar tool and packing source "
                    + "files to Jar.</p>"
            ),
            @Param(
                name = "host_javabase",
                type = ConfiguredTarget.class,
                named = true,
                defaultValue = "None",
                noneable = true,
                doc = "Deprecated. No longer needed when <code>compile_jar</code> and/or "
                    + "<code>source_jar</code> are used. "
                    + "<p>The host_javabase to be used for packing source files to Jar.</p>"
            ),
        },
        selfCall = true,
        useLocation = true,
        useEnvironment = true
    )
    @SkylarkConstructor(
        objectType = JavaInfo.class,
        receiverNameForDoc = "JavaInfo"
    )
    @SuppressWarnings({"unchecked"})
    public JavaInfo javaInfo(
        FileApi outputJarApi,
        Object compileJarApi,
        Object sourceJarApi,
        Boolean neverlink,
        SkylarkList<?> deps,
        SkylarkList<?> runtimeDeps,
        SkylarkList<?> exports,
        Object actionsApi,
        Object sourcesApi,
        Object sourceJarsApi,
        Object useIjarApi,
        Object javaToolchainApi,
        Object hostJavabaseApi,
        Location loc,
        Environment env) throws EvalException {
      Artifact outputJar = (Artifact) outputJarApi;
      @Nullable Artifact compileJar = nullIfNone(compileJarApi, Artifact.class);
      @Nullable Artifact sourceJar = nullIfNone(sourceJarApi, Artifact.class);

      @Nullable Object actions = nullIfNone(actionsApi);
      @Nullable SkylarkList<Artifact> sources =
          (SkylarkList<Artifact>) nullIfNone(sourcesApi, SkylarkList.class);
      @Nullable
      SkylarkList<Artifact> sourceJars =
          (SkylarkList<Artifact>) nullIfNone(sourceJarsApi, SkylarkList.class);

      @Nullable Boolean useIjar = nullIfNone(useIjarApi, Boolean.class);
      @Nullable Object javaToolchain = nullIfNone(javaToolchainApi);
      @Nullable Object hostJavabase = nullIfNone(hostJavabaseApi);

      boolean hasLegacyArg =
          actions != null
              || sources != null
              || sourceJars != null
              || useIjar != null
              || javaToolchain != null
              || hostJavabase != null;
      if (hasLegacyArg) {
        if (env.getSemantics().incompatibleDisallowLegacyJavaInfo()) {
          throw new EvalException(
              loc,
              "Cannot use deprecated argument when "
                  + "--incompatible_disallow_legacy_javainfo is set. "
                  + "Deprecated arguments are 'actions', 'sources', 'source_jars', "
                  + "'use_ijar', 'java_toolchain', 'host_javabase'.");
        }
        boolean hasNewArg = compileJar != null || sourceJar != null;
        if (hasNewArg) {
          throw new EvalException(
              loc,
              "Cannot use deprecated arguments at the same time as "
                  + "'compile_jar' or 'source_jar'. "
                  + "Deprecated arguments are 'actions', 'sources', 'source_jars', "
                  + "'use_ijar', 'java_toolchain', 'host_javabase'.");
        }
        return JavaInfoBuildHelper.getInstance()
            .createJavaInfoLegacy(
                outputJar,
                sources != null ? sources : MutableList.empty(),
                sourceJars != null ? sourceJars : MutableList.empty(),
                useIjar != null ? useIjar : true,
                neverlink,
                (SkylarkList<JavaInfo>) deps,
                (SkylarkList<JavaInfo>) runtimeDeps,
                (SkylarkList<JavaInfo>) exports,
                actions,
                javaToolchain,
                hostJavabase,
                loc);
      }
      if (compileJar == null) {
        throw new EvalException(loc, "Expected 'File' for 'compile_jar', found 'None'");
      }
      return JavaInfoBuildHelper.getInstance()
          .createJavaInfo(
              outputJar, compileJar, sourceJar, neverlink,
              (SkylarkList<JavaInfo>) deps,
              (SkylarkList<JavaInfo>) runtimeDeps,
              (SkylarkList<JavaInfo>) exports, loc);
    }
  }

  /**
   * A Builder for {@link JavaInfo}.
   */
  public static class Builder {
    TransitiveInfoProviderMapBuilder providerMap;
    private ImmutableList<Artifact> runtimeJars;
    private ImmutableList<String> javaConstraints;
    private boolean neverlink;
    private Location location = Location.BUILTIN;

    private Builder(TransitiveInfoProviderMapBuilder providerMap) {
      this.providerMap = providerMap;
    }

    public static Builder create() {
      return new Builder(new TransitiveInfoProviderMapBuilder())
          .setRuntimeJars(ImmutableList.of())
          .setJavaConstraints(ImmutableList.of());
    }

    public static Builder copyOf(JavaInfo javaInfo) {
      return new Builder(new TransitiveInfoProviderMapBuilder().addAll(javaInfo.getProviders()))
          .setRuntimeJars(javaInfo.getDirectRuntimeJars())
          .setNeverlink(javaInfo.isNeverlink())
          .setJavaConstraints(javaInfo.getJavaConstraints())
          .setLocation(javaInfo.getCreationLoc());
    }

    public Builder setRuntimeJars(ImmutableList<Artifact> runtimeJars) {
      this.runtimeJars = runtimeJars;
      return this;
    }

    public Builder setNeverlink(boolean neverlink) {
      this.neverlink = neverlink;
      return this;
    }

    public Builder setJavaConstraints(ImmutableList<String> javaConstraints) {
      this.javaConstraints = javaConstraints;
      return this;
    }

    public Builder setLocation(Location location) {
      this.location = location;
      return this;
    }

    public <P extends TransitiveInfoProvider> Builder addProvider(
        Class<P> providerClass, TransitiveInfoProvider provider) {
      Preconditions.checkArgument(ALLOWED_PROVIDERS.contains(providerClass));
      providerMap.put(providerClass, provider);
      return this;
    }

    public JavaInfo build() {
      return new JavaInfo(providerMap.build(), runtimeJars, neverlink, javaConstraints, location);
    }
  }
}

