// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.skylark.SkylarkErrorReporter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarInfo.Aar;
import com.google.devtools.build.lib.rules.android.databinding.DataBinding;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidBinaryDataSettingsApi;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidDataProcessingApi;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;

/** Skylark-visible methods for working with Android data (manifests, resources, and assets). */
public abstract class AndroidSkylarkData
    implements AndroidDataProcessingApi<
        AndroidDataContext,
        ConfiguredTarget,
        Artifact,
        SpecialArtifact,
        AndroidAssetsInfo,
        AndroidResourcesInfo,
        AndroidManifestInfo,
        AndroidLibraryAarInfo,
        AndroidBinaryDataInfo,
        ValidatedAndroidResources> {

  public abstract AndroidSemantics getAndroidSemantics();

  @Override
  public AndroidAssetsInfo assetsFromDeps(
      Sequence<?> deps, // <AndroidAssetsInfo>
      boolean neverlink,
      StarlarkThread thread)
      throws EvalException {
    // We assume this is an analysis-phase thread.
    Label label = BazelStarlarkContext.from(thread).getAnalysisRuleLabel();
    return AssetDependencies.fromProviders(
            deps.getContents(AndroidAssetsInfo.class, "deps"), neverlink)
        .toInfo(label);
  }

  @Override
  public AndroidResourcesInfo resourcesFromDeps(
      AndroidDataContext ctx,
      Sequence<?> deps, // <AndroidResourcesInfo>
      Sequence<?> assets, // <AndroidAssetsInfo>
      boolean neverlink,
      String customPackage,
      Location location,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    try (SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location)) {
      return ResourceApk.processFromTransitiveLibraryData(
              ctx,
              DataBinding.getDisabledDataBindingContext(ctx),
              ResourceDependencies.fromProviders(
                  deps.getContents(AndroidResourcesInfo.class, "deps"),
                  /* neverlink = */ neverlink),
              AssetDependencies.fromProviders(
                  assets.getContents(AndroidAssetsInfo.class, "assets"),
                  /* neverlink = */ neverlink),
              StampedAndroidManifest.createEmpty(
                  ctx.getActionConstructionContext(), customPackage, /* exported = */ false))
          .toResourceInfo(ctx.getLabel());
    }
  }

  @Override
  public AndroidManifestInfo stampAndroidManifest(
      AndroidDataContext ctx,
      Object manifest,
      Object customPackage,
      boolean exported,
      Location location,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    String pkg = fromNoneable(customPackage, String.class);
    try (SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location)) {
      return AndroidManifest.from(
              ctx,
              errorReporter,
              fromNoneable(manifest, Artifact.class),
              getAndroidSemantics(),
              pkg,
              exported)
          .stamp(ctx)
          .toProvider();
    }
  }

  @Override
  public AndroidAssetsInfo mergeAssets(
      AndroidDataContext ctx,
      Object assets,
      Object assetsDir,
      Sequence<?> deps, // <AndroidAssetsInfo>
      boolean neverlink,
      Location location,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location);
    try {
      return AndroidAssets.from(
              errorReporter,
              listFromNoneable(assets, ConfiguredTarget.class),
              isNone(assetsDir) ? null : PathFragment.create(fromNoneable(assetsDir, String.class)))
          .process(
              ctx,
              AssetDependencies.fromProviders(
                  deps.getContents(AndroidAssetsInfo.class, "deps"), neverlink))
          .toProvider();
    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
  }

  @Override
  public ValidatedAndroidResources mergeRes(
      AndroidDataContext ctx,
      AndroidManifestInfo manifest,
      Sequence<?> resources, // <ConfiguredTarget>
      Sequence<?> deps, // <AndroidResourcesInfo>
      boolean neverlink,
      boolean enableDataBinding,
      Location location,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location);
    try {
      return AndroidResources.from(
              errorReporter,
              getFileProviders(resources.getContents(ConfiguredTarget.class, "resources")),
              "resources")
          .process(
              ctx,
              manifest.asStampedManifest(),
              ResourceDependencies.fromProviders(
                  deps.getContents(AndroidResourcesInfo.class, "deps"), neverlink),
              DataBinding.contextFrom(
                  enableDataBinding, ctx.getActionConstructionContext(), ctx.getAndroidConfig()));
    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
  }

  @Override
  public Dict<Provider, NativeInfo> mergeResources(
      AndroidDataContext ctx,
      AndroidManifestInfo manifest,
      Sequence<?> resources, // <ConfiguredTarget>
      Sequence<?> deps, // <AndroidResourcesInfo>
      boolean neverlink,
      boolean enableDataBinding,
      Location location,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    ValidatedAndroidResources validated =
        mergeRes(ctx, manifest, resources, deps, neverlink, enableDataBinding, location, thread);
    JavaInfo javaInfo =
        getJavaInfoForRClassJar(validated.getClassJar(), validated.getJavaSourceJar());
    return Dict.of(
        (Mutability) null,
        AndroidResourcesInfo.PROVIDER,
        validated.toProvider(),
        JavaInfo.PROVIDER,
        javaInfo);
  }

  @Override
  public AndroidLibraryAarInfo makeAar(
      AndroidDataContext ctx,
      AndroidResourcesInfo resourcesInfo,
      AndroidAssetsInfo assetsInfo,
      Artifact libraryClassJar,
      Sequence<?> localProguardSpecs, // <Artifact>
      Sequence<?> deps, // <AndroidLibraryAarInfo>
      boolean neverlink)
      throws EvalException, InterruptedException {
    if (neverlink) {
      return AndroidLibraryAarInfo.create(
          null,
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER));
    }

    // Get the target's local resources, if defined, from the provider
    Optional<? extends AndroidResources> resources =
        resourcesInfo.getDirectAndroidResources().toList().stream()
            .filter(r -> r.getLabel().equals(ctx.getLabel()))
            .findFirst();
    boolean definesLocalResources = resources.isPresent();

    // Get the target's local assets, if defined, from the provider
    Optional<? extends AndroidAssets> assets =
        assetsInfo.getDirectParsedAssets().toList().stream()
            .filter(a -> a.getLabel().equals(ctx.getLabel()))
            .findFirst();
    // The target might still define an empty list of assets, in which case its information is not
    // propagated for efficiency. If this is the case, we will still have an artifact for the
    // merging output.
    boolean definesLocalAssets = assets.isPresent() || assetsInfo.getValidationResult() != null;

    if (definesLocalResources != definesLocalAssets) {
      throw new EvalException(
          Location.BUILTIN,
          "Must define either both or none of assets and resources. Use the merge_assets and"
              + " merge_resources methods to define them, or assets_from_deps and"
              + " resources_from_deps to inherit without defining them.");
    }

    return Aar.makeAar(
            ctx,
            resources.isPresent() ? resources.get() : AndroidResources.empty(),
            assets.isPresent() ? assets.get() : AndroidAssets.empty(),
            resourcesInfo.getManifest(),
            resourcesInfo.getRTxt(),
            libraryClassJar,
            ImmutableList.copyOf(
                localProguardSpecs.getContents(Artifact.class, "local_proguard_specs")))
        .toProvider(deps.getContents(AndroidLibraryAarInfo.class, "deps"), definesLocalResources);
  }

  @Override
  public Dict<Provider, NativeInfo> processAarImportData(
      AndroidDataContext ctx,
      SpecialArtifact resources,
      SpecialArtifact assets,
      Artifact androidManifestArtifact,
      Sequence<?> deps) // <ConfiguredTarget>
      throws InterruptedException, EvalException {
    List<ConfiguredTarget> depsTargets = deps.getContents(ConfiguredTarget.class, "deps");

    ValidatedAndroidResources validatedResources =
        AndroidResources.forAarImport(resources)
            .process(
                ctx,
                AndroidManifest.forAarImport(androidManifestArtifact),
                ResourceDependencies.fromProviders(
                    getProviders(depsTargets, AndroidResourcesInfo.PROVIDER),
                    /* neverlink = */ false),
                DataBinding.getDisabledDataBindingContext(ctx));

    MergedAndroidAssets mergedAssets =
        AndroidAssets.forAarImport(assets)
            .process(
                ctx,
                AssetDependencies.fromProviders(
                    getProviders(depsTargets, AndroidAssetsInfo.PROVIDER),
                    /* neverlink = */ false));

    ResourceApk resourceApk = ResourceApk.of(validatedResources, mergedAssets, null, null);

    return getNativeInfosFrom(resourceApk, ctx.getLabel());
  }

  @Override
  public Dict<Provider, NativeInfo> processLocalTestData(
      AndroidDataContext ctx,
      Object manifest,
      Sequence<?> resources, // <ConfiguredTarget>
      Object assets,
      Object assetsDir,
      Object customPackage,
      String aaptVersionString,
      Dict<?, ?> manifestValues, // <String, String>
      Sequence<?> deps, // <ConfiguredTarget>
      Sequence<?> noCompressExtensions, // <String>
      Location location,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location);
    List<ConfiguredTarget> depsTargets = deps.getContents(ConfiguredTarget.class, "deps");

    try {
      AndroidManifest rawManifest =
          AndroidManifest.from(
              ctx,
              errorReporter,
              fromNoneable(manifest, Artifact.class),
              fromNoneable(customPackage, String.class),
              /* exportsManifest = */ false);

      ResourceApk resourceApk =
          AndroidLocalTestBase.buildResourceApk(
              ctx,
              getAndroidSemantics(),
              errorReporter,
              DataBinding.getDisabledDataBindingContext(ctx),
              rawManifest,
              AndroidResources.from(
                  errorReporter,
                  getFileProviders(resources.getContents(ConfiguredTarget.class, "resource_files")),
                  "resource_files"),
              AndroidAssets.from(
                  errorReporter,
                  listFromNoneable(assets, ConfiguredTarget.class),
                  isNone(assetsDir)
                      ? null
                      : PathFragment.create(fromNoneable(assetsDir, String.class))),
              ResourceDependencies.fromProviders(
                  getProviders(depsTargets, AndroidResourcesInfo.PROVIDER),
                  /* neverlink = */ false),
              AssetDependencies.fromProviders(
                  getProviders(depsTargets, AndroidAssetsInfo.PROVIDER), /* neverlink = */ false),
              manifestValues.getContents(String.class, String.class, "manifest_values"),
              noCompressExtensions.getContents(String.class, "nocompress_extensions"));

      ImmutableMap.Builder<Provider, NativeInfo> builder = ImmutableMap.builder();
      builder.putAll(getNativeInfosFrom(resourceApk, ctx.getLabel()));
      builder.put(
          AndroidBinaryDataInfo.PROVIDER,
          AndroidBinaryDataInfo.of(
              resourceApk.getArtifact(),
              resourceApk.getResourceProguardConfig(),
              resourceApk.toResourceInfo(ctx.getLabel()),
              resourceApk.toAssetsInfo(ctx.getLabel()),
              resourceApk.toManifestInfo().get()));
      return Dict.copyOf((Mutability) null, builder.build());
    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
  }

  private static IllegalStateException handleRuleException(
      SkylarkErrorReporter errorReporter, RuleErrorException exception) throws EvalException {
    // The error reporter should have been notified of the rule error, and thus closing it will
    // throw an EvalException.
    errorReporter.close();
    // It's a catastrophic state error if the errorReporter did not pick up the error.
    throw new IllegalStateException("Unhandled RuleErrorException", exception);
  }

  @Override
  public BinaryDataSettings makeBinarySettings(
      AndroidDataContext ctx,
      Object shrinkResources,
      Sequence<?> resourceConfigurationFilters, // <String>
      Sequence<?> densities, // <String>
      Sequence<?> noCompressExtensions, // <String>
      Location location,
      StarlarkThread thread)
      throws EvalException {
    return new BinaryDataSettings(
        fromNoneableOrDefault(
            shrinkResources, Boolean.class, ctx.getAndroidConfig().useAndroidResourceShrinking()),
        ResourceFilterFactory.from(
            resourceConfigurationFilters.getContents(
                String.class, "resource_configuration_filters"),
            densities.getContents(String.class, "densities")),
        ImmutableList.copyOf(
            noCompressExtensions.getContents(String.class, "nocompress_extensions")));
  }

  @Override
  public Artifact resourcesFromValidatedRes(ValidatedAndroidResources resources) {
    return resources.getMergedResources();
  }

  /**
   * Helper method to get default {@link
   * com.google.devtools.build.lib.rules.android.AndroidSkylarkData.BinaryDataSettings}.
   */
  private BinaryDataSettings defaultBinaryDataSettings(
      AndroidDataContext ctx, Location location, StarlarkThread thread) throws EvalException {
    return makeBinarySettings(
        ctx,
        Starlark.NONE,
        StarlarkList.empty(),
        StarlarkList.empty(),
        StarlarkList.empty(),
        location,
        thread);
  }

  private static class BinaryDataSettings implements AndroidBinaryDataSettingsApi {
    private final boolean shrinkResources;
    private final ResourceFilterFactory resourceFilterFactory;
    private final ImmutableList<String> noCompressExtensions;

    private BinaryDataSettings(
        boolean shrinkResources,
        ResourceFilterFactory resourceFilterFactory,
        ImmutableList<String> noCompressExtensions) {
      this.shrinkResources = shrinkResources;
      this.resourceFilterFactory = resourceFilterFactory;
      this.noCompressExtensions = noCompressExtensions;
    }
  }

  @Override
  public AndroidBinaryDataInfo processBinaryData(
      AndroidDataContext ctx,
      Sequence<?> resources,
      Object assets,
      Object assetsDir,
      Object manifest,
      Object customPackage,
      Dict<?, ?> manifestValues, // <String, String>
      Sequence<?> deps, // <ConfiguredTarget>
      String manifestMerger,
      Object maybeSettings,
      boolean crunchPng,
      boolean dataBindingEnabled,
      Location location,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location);
    List<ConfiguredTarget> depsTargets = deps.getContents(ConfiguredTarget.class, "deps");
    Map<String, String> manifestValueMap =
        manifestValues.getContents(String.class, String.class, "manifest_values");

    try {
      BinaryDataSettings settings =
          fromNoneableOrDefault(
              maybeSettings,
              BinaryDataSettings.class,
              defaultBinaryDataSettings(ctx, location, thread));

      AndroidManifest rawManifest =
          AndroidManifest.from(
              ctx,
              errorReporter,
              fromNoneable(manifest, Artifact.class),
              getAndroidSemantics(),
              fromNoneable(customPackage, String.class),
              /* exportsManifest = */ false);

      ResourceDependencies resourceDeps =
          ResourceDependencies.fromProviders(
              getProviders(depsTargets, AndroidResourcesInfo.PROVIDER), /* neverlink = */ false);

      StampedAndroidManifest stampedManifest =
          rawManifest.mergeWithDeps(
              ctx,
              getAndroidSemantics(),
              errorReporter,
              resourceDeps,
              manifestValueMap,
              manifestMerger);

      ResourceApk resourceApk =
          ProcessedAndroidData.processBinaryDataFrom(
                  ctx,
                  errorReporter,
                  stampedManifest,
                  AndroidBinary.shouldShrinkResourceCycles(
                      ctx.getAndroidConfig(), errorReporter, settings.shrinkResources),
                  manifestValueMap,
                  AndroidResources.from(
                      errorReporter,
                      getFileProviders(
                          resources.getContents(ConfiguredTarget.class, "resource_files")),
                      "resource_files"),
                  AndroidAssets.from(
                      errorReporter,
                      listFromNoneable(assets, ConfiguredTarget.class),
                      isNone(assetsDir)
                          ? null
                          : PathFragment.create(fromNoneable(assetsDir, String.class))),
                  resourceDeps,
                  AssetDependencies.fromProviders(
                      getProviders(depsTargets, AndroidAssetsInfo.PROVIDER),
                      /* neverlink = */ false),
                  settings.resourceFilterFactory,
                  settings.noCompressExtensions,
                  crunchPng,
                  DataBinding.contextFrom(
                      dataBindingEnabled,
                      ctx.getActionConstructionContext(),
                      ctx.getAndroidConfig()))
              .generateRClass(ctx);

      return AndroidBinaryDataInfo.of(
          resourceApk.getArtifact(),
          resourceApk.getResourceProguardConfig(),
          resourceApk.toResourceInfo(ctx.getLabel()),
          resourceApk.toAssetsInfo(ctx.getLabel()),
          resourceApk.toManifestInfo().get());

    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
  }

  @Override
  public AndroidBinaryDataInfo shrinkDataApk(
      AndroidDataContext ctx,
      AndroidBinaryDataInfo binaryDataInfo,
      Artifact proguardOutputJar,
      Artifact proguardMapping,
      Object maybeSettings,
      Sequence<?> deps, // <ConfiguredTarget>
      Sequence<?> localProguardSpecs, // <ConfiguredTarget>
      Sequence<?> extraProguardSpecs, // <ConfiguredTarget>
      Location location,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    BinaryDataSettings settings =
        fromNoneableOrDefault(
            maybeSettings,
            BinaryDataSettings.class,
            defaultBinaryDataSettings(ctx, location, thread));
    List<ConfiguredTarget> depsTargets = deps.getContents(ConfiguredTarget.class, "deps");

    if (!settings.shrinkResources) {
      return binaryDataInfo;
    }

    ImmutableList<Artifact> proguardSpecs =
        AndroidBinary.getProguardSpecs(
            ctx,
            getAndroidSemantics(),
            binaryDataInfo.getResourceProguardConfig(),
            binaryDataInfo.getManifestInfo().getManifest(),
            filesFromConfiguredTargets(
                localProguardSpecs.getContents(ConfiguredTarget.class, "proguard_specs")),
            filesFromConfiguredTargets(
                extraProguardSpecs.getContents(ConfiguredTarget.class, "extra_proguard_specs")),
            getProviders(depsTargets, ProguardSpecProvider.PROVIDER));

    // TODO(asteinb): There should never be more than one direct resource exposed in the provider.
    // Can we adjust its structure to take this into account?
    if (!binaryDataInfo.getResourcesInfo().getDirectAndroidResources().isSingleton()) {
      throw new EvalException(
          Location.BUILTIN,
          "Expected exactly 1 direct android resource container, but found: "
              + binaryDataInfo.getResourcesInfo().getDirectAndroidResources());
    }

    if (!proguardSpecs.isEmpty()) {
      Artifact shrunkApk =
          AndroidBinary.shrinkResources(
              ctx,
              binaryDataInfo.getResourcesInfo().getDirectAndroidResources().toList().get(0),
              ResourceDependencies.fromProviders(
                  getProviders(depsTargets, AndroidResourcesInfo.PROVIDER),
                  /* neverlink = */ false),
              proguardOutputJar,
              proguardMapping,
              settings.resourceFilterFactory,
              settings.noCompressExtensions);
      return binaryDataInfo.withShrunkApk(shrunkApk);
    }

    return binaryDataInfo;
  }

  public static Dict<Provider, NativeInfo> getNativeInfosFrom(
      ResourceApk resourceApk, Label label) {
    ImmutableMap.Builder<Provider, NativeInfo> builder = ImmutableMap.builder();

    builder
        .put(AndroidResourcesInfo.PROVIDER, resourceApk.toResourceInfo(label))
        .put(AndroidAssetsInfo.PROVIDER, resourceApk.toAssetsInfo(label));

    resourceApk.toManifestInfo().ifPresent(info -> builder.put(AndroidManifestInfo.PROVIDER, info));

    builder.put(
        JavaInfo.PROVIDER,
        getJavaInfoForRClassJar(
            resourceApk.getResourceJavaClassJar(), resourceApk.getResourceJavaSrcJar()));

    return Dict.copyOf((Mutability) null, builder.build());
  }

  private static JavaInfo getJavaInfoForRClassJar(Artifact rClassJar, Artifact rClassSrcJar) {
    return JavaInfo.Builder.create()
        .setNeverlink(true)
        .addProvider(
            JavaSourceJarsProvider.class,
            JavaSourceJarsProvider.builder().addSourceJar(rClassSrcJar).build())
        .addProvider(
            JavaRuleOutputJarsProvider.class,
            JavaRuleOutputJarsProvider.builder()
                .addOutputJar(rClassJar, null, null, ImmutableList.of(rClassSrcJar))
                .build())
        .addProvider(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider.builder()
                .addDirectCompileTimeJar(rClassJar, rClassJar)
                .build())
        .addProvider(
            JavaCompilationInfoProvider.class,
            new JavaCompilationInfoProvider.Builder()
                .setCompilationClasspath(NestedSetBuilder.create(Order.NAIVE_LINK_ORDER, rClassJar))
                .build())
        .build();
  }

  /** Checks if a "Noneable" object passed by Skylark is "None", which Java should treat as null. */
  public static boolean isNone(Object object) {
    return object == Starlark.NONE;
  }

  /**
   * Converts a "Noneable" Object passed by Skylark to an nullable object of the appropriate type.
   *
   * <p>Skylark "Noneable" types are passed in as an Object that may be either the correct type or a
   * Starlark.NONE object. Skylark will handle type checking, based on the appropriate @param
   * annotation, but we still need to do the actual cast (or conversion to null) ourselves.
   *
   * @param object the Noneable object
   * @param clazz the correct class, as defined in the @Param annotation
   * @param <T> the type to cast to
   * @return {@code null}, if the noneable argument was None, or the cast object, otherwise.
   */
  @Nullable
  public static <T> T fromNoneable(Object object, Class<T> clazz) {
    if (isNone(object)) {
      return null;
    }

    return clazz.cast(object);
  }

  public static <T> T fromNoneableOrDefault(Object object, Class<T> clazz, T defaultValue) {
    T value = fromNoneable(object, clazz);
    if (value == null) {
      return defaultValue;
    }

    return value;
  }

  /**
   * Converts a "Noneable" Object passed by Skylark to a List of the appropriate type.
   *
   * <p>This first calls {@link #fromNoneable(Object, Class)} to get a Sequence<?>, then safely
   * casts it to a list with the appropriate generic.
   */
  @Nullable
  public static <T> List<T> listFromNoneable(Object object, Class<T> clazz) throws EvalException {
    Sequence<?> asList = fromNoneable(object, Sequence.class);
    if (asList == null) {
      return null;
    }

    return Sequence.castList(asList, clazz, null);
  }

  private static ImmutableList<Artifact> filesFromConfiguredTargets(
      List<ConfiguredTarget> targets) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (FileProvider provider : getFileProviders(targets)) {
      builder.addAll(provider.getFilesToBuild().toList());
    }

    return builder.build();
  }

  private static ImmutableList<FileProvider> getFileProviders(List<ConfiguredTarget> targets) {
    return getProviders(targets, FileProvider.class);
  }

  private static <T extends TransitiveInfoProvider> ImmutableList<T> getProviders(
      List<ConfiguredTarget> targets, Class<T> clazz) {
    return targets
        .stream()
        .map(target -> target.getProvider(clazz))
        .filter(Objects::nonNull)
        .collect(ImmutableList.toImmutableList());
  }

  public static <T extends NativeInfo> Sequence<T> getProviders(
      List<ConfiguredTarget> targets, NativeProvider<T> provider) {
    return StarlarkList.immutableCopyOf(
        targets.stream()
            .map(target -> target.get(provider))
            .filter(Objects::nonNull)
            .collect(ImmutableList.toImmutableList()));
  }

  protected static <T extends NativeInfo> Sequence<T> getProviders(
      List<ConfiguredTarget> targets, BuiltinProvider<T> provider) {
    return StarlarkList.immutableCopyOf(
        targets.stream()
            .map(target -> target.get(provider))
            .filter(Objects::nonNull)
            .collect(ImmutableList.toImmutableList()));
  }
}
