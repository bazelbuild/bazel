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
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
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
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
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
      SkylarkList<AndroidAssetsInfo> deps, boolean neverlink, Environment env) {
    return AssetDependencies.fromProviders(deps, neverlink).toInfo(env.getCallerLabel());
  }

  @Override
  public AndroidResourcesInfo resourcesFromDeps(
      AndroidDataContext ctx,
      SkylarkList<AndroidResourcesInfo> deps,
      SkylarkList<AndroidAssetsInfo> assets,
      boolean neverlink,
      String customPackage,
      Location location,
      Environment env)
      throws InterruptedException, EvalException {
    try (SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location)) {
      return ResourceApk.processFromTransitiveLibraryData(
              ctx,
              DataBinding.getDisabledDataBindingContext(ctx),
              ResourceDependencies.fromProviders(deps, /* neverlink = */ neverlink),
              AssetDependencies.fromProviders(assets, /* neverlink = */ neverlink),
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
      Environment env)
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
      SkylarkList<AndroidAssetsInfo> deps,
      boolean neverlink,
      Location location,
      Environment env)
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
              AssetDependencies.fromProviders(deps, neverlink),
              getAndroidAaptVersionForLibrary(ctx))
          .toProvider();
    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
  }

  @Override
  public ValidatedAndroidResources mergeRes(
      AndroidDataContext ctx,
      AndroidManifestInfo manifest,
      SkylarkList<ConfiguredTarget> resources,
      SkylarkList<AndroidResourcesInfo> deps,
      boolean neverlink,
      boolean enableDataBinding,
      Location location,
      Environment env)
      throws EvalException, InterruptedException {
    SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location);
    try {
      return AndroidResources.from(errorReporter, getFileProviders(resources), "resources")
          .process(
              ctx,
              manifest.asStampedManifest(),
              ResourceDependencies.fromProviders(deps, neverlink),
              DataBinding.contextFrom(
                  enableDataBinding, ctx.getActionConstructionContext(), ctx.getAndroidConfig()),
              getAndroidAaptVersionForLibrary(ctx));
    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
  }

  @Override
  public SkylarkDict<Provider, NativeInfo> mergeResources(
      AndroidDataContext ctx,
      AndroidManifestInfo manifest,
      SkylarkList<ConfiguredTarget> resources,
      SkylarkList<AndroidResourcesInfo> deps,
      boolean neverlink,
      boolean enableDataBinding,
      Location location,
      Environment env)
      throws EvalException, InterruptedException {
    ValidatedAndroidResources validated =
        mergeRes(ctx, manifest, resources, deps, neverlink, enableDataBinding, location, env);
    JavaInfo javaInfo =
        getJavaInfoForRClassJar(validated.getClassJar(), validated.getJavaSourceJar());
    return SkylarkDict.of(
        /* env = */ null,
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
      SkylarkList<Artifact> localProguardSpecs,
      SkylarkList<AndroidLibraryAarInfo> deps,
      boolean neverlink)
      throws EvalException, InterruptedException {
    if (neverlink) {
      return AndroidLibraryAarInfo.create(
          null,
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
          NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER));
    }

    // Get the target's local resources, if defined, from the provider
    boolean definesLocalResources = resourcesInfo.getDirectAndroidResources().isSingleton();
    AndroidResources resources = AndroidResources.empty();
    if (definesLocalResources) {
      ValidatedAndroidResources validatedAndroidResources =
          resourcesInfo.getDirectAndroidResources().toList().get(0);
      if (validatedAndroidResources.getLabel().equals(ctx.getLabel())) {
        resources = validatedAndroidResources;
      } else {
        definesLocalResources = false;
      }
    }

    // Get the target's local assets, if defined, from the provider
    boolean definesLocalAssets = false;
    AndroidAssets assets = AndroidAssets.empty();
    if (assetsInfo.getDirectParsedAssets().isSingleton()) {
      ParsedAndroidAssets parsed = assetsInfo.getDirectParsedAssets().toList().get(0);
      if (parsed.getLabel().equals(ctx.getLabel())) {
        assets = parsed;
        definesLocalAssets = true;
      }
    }

    if (!definesLocalAssets) {
      // The target might still define an empty list of assets, in which case its information is not
      // propagated for efficiency. If this is the case, we will still have an artifact for the
      // merging output.
      definesLocalAssets = assetsInfo.getValidationResult() != null;
    }

    if (definesLocalResources != definesLocalAssets) {
      throw new EvalException(
          Location.BUILTIN,
          "Must define either both or none of assets and resources. Use the merge_assets and"
              + " merge_resources methods to define them, or assets_from_deps and"
              + " resources_from_deps to inherit without defining them.");
    }

    return Aar.makeAar(
            ctx,
            resources,
            assets,
            resourcesInfo.getManifest(),
            resourcesInfo.getRTxt(),
            libraryClassJar,
            localProguardSpecs.getImmutableList())
        .toProvider(deps, definesLocalResources);
  }

  @Override
  public SkylarkDict<Provider, NativeInfo> processAarImportData(
      AndroidDataContext ctx,
      SpecialArtifact resources,
      SpecialArtifact assets,
      Artifact androidManifestArtifact,
      SkylarkList<ConfiguredTarget> deps)
      throws InterruptedException {

    AndroidAaptVersion aaptVersion = getAndroidAaptVersionForLibrary(ctx);

    ValidatedAndroidResources validatedResources =
        AndroidResources.forAarImport(resources)
            .process(
                ctx,
                AndroidManifest.forAarImport(androidManifestArtifact),
                ResourceDependencies.fromProviders(
                    getProviders(deps, AndroidResourcesInfo.PROVIDER), /* neverlink = */ false),
                DataBinding.getDisabledDataBindingContext(ctx),
                aaptVersion);

    MergedAndroidAssets mergedAssets =
        AndroidAssets.forAarImport(assets)
            .process(
                ctx,
                AssetDependencies.fromProviders(
                    getProviders(deps, AndroidAssetsInfo.PROVIDER), /* neverlink = */ false),
                aaptVersion);

    ResourceApk resourceApk = ResourceApk.of(validatedResources, mergedAssets, null, null);

    return getNativeInfosFrom(resourceApk, ctx.getLabel());
  }

  @Override
  public SkylarkDict<Provider, NativeInfo> processLocalTestData(
      AndroidDataContext ctx,
      Object manifest,
      SkylarkList<ConfiguredTarget> resources,
      Object assets,
      Object assetsDir,
      Object customPackage,
      String aaptVersionString,
      SkylarkDict<String, String> manifestValues,
      SkylarkList<ConfiguredTarget> deps,
      SkylarkList<String> noCompressExtensions,
      Location location,
      Environment env)
      throws InterruptedException, EvalException {
    SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location);

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
              AndroidResources.from(errorReporter, getFileProviders(resources), "resource_files"),
              AndroidAssets.from(
                  errorReporter,
                  listFromNoneable(assets, ConfiguredTarget.class),
                  isNone(assetsDir)
                      ? null
                      : PathFragment.create(fromNoneable(assetsDir, String.class))),
              ResourceDependencies.fromProviders(
                  getProviders(deps, AndroidResourcesInfo.PROVIDER), /* neverlink = */ false),
              AssetDependencies.fromProviders(
                  getProviders(deps, AndroidAssetsInfo.PROVIDER), /* neverlink = */ false),
              manifestValues,
              AndroidAaptVersion.chooseTargetAaptVersion(ctx, errorReporter, aaptVersionString),
              noCompressExtensions);

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
      return SkylarkDict.copyOf(/* env = */ null, builder.build());
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
      SkylarkList<String> resourceConfigurationFilters,
      SkylarkList<String> densities,
      SkylarkList<String> noCompressExtensions,
      String aaptVersionString,
      Location location,
      Environment env)
      throws EvalException {

    SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location);
    AndroidAaptVersion aaptVersion;

    try {
      aaptVersion =
          AndroidAaptVersion.chooseTargetAaptVersion(ctx, errorReporter, aaptVersionString);

      return new BinaryDataSettings(
          aaptVersion,
          fromNoneableOrDefault(
              shrinkResources, Boolean.class, ctx.getAndroidConfig().useAndroidResourceShrinking()),
          ResourceFilterFactory.from(aaptVersion, resourceConfigurationFilters, densities),
          noCompressExtensions.getImmutableList());
    } catch (RuleErrorException e) {
      throw handleRuleException(errorReporter, e);
    }
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
      AndroidDataContext ctx, Location location, Environment env) throws EvalException {
    return makeBinarySettings(
        ctx,
        Runtime.NONE,
        SkylarkList.createImmutable(ImmutableList.of()),
        SkylarkList.createImmutable(ImmutableList.of()),
        SkylarkList.createImmutable(ImmutableList.of()),
        "auto",
        location,
        env);
  }

  private static class BinaryDataSettings implements AndroidBinaryDataSettingsApi {
    private final AndroidAaptVersion aaptVersion;
    private final boolean shrinkResources;
    private final ResourceFilterFactory resourceFilterFactory;
    private final ImmutableList<String> noCompressExtensions;

    private BinaryDataSettings(
        AndroidAaptVersion aaptVersion,
        boolean shrinkResources,
        ResourceFilterFactory resourceFilterFactory,
        ImmutableList<String> noCompressExtensions) {
      this.aaptVersion = aaptVersion;
      this.shrinkResources = shrinkResources;
      this.resourceFilterFactory = resourceFilterFactory;
      this.noCompressExtensions = noCompressExtensions;
    }
  }

  @Override
  public AndroidBinaryDataInfo processBinaryData(
      AndroidDataContext ctx,
      SkylarkList<ConfiguredTarget> resources,
      Object assets,
      Object assetsDir,
      Object manifest,
      Object customPackage,
      SkylarkDict<String, String> manifestValues,
      SkylarkList<ConfiguredTarget> deps,
      String manifestMerger,
      Object maybeSettings,
      boolean crunchPng,
      boolean dataBindingEnabled,
      Location location,
      Environment env)
      throws InterruptedException, EvalException {
    SkylarkErrorReporter errorReporter =
        SkylarkErrorReporter.from(ctx.getRuleErrorConsumer(), location);

    try {
      BinaryDataSettings settings =
          fromNoneableOrDefault(
              maybeSettings,
              BinaryDataSettings.class,
              defaultBinaryDataSettings(ctx, location, env));

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
              getProviders(deps, AndroidResourcesInfo.PROVIDER), /* neverlink = */ false);

      StampedAndroidManifest stampedManifest =
          rawManifest.mergeWithDeps(
              ctx,
              getAndroidSemantics(),
              errorReporter,
              resourceDeps,
              manifestValues,
              manifestMerger);

      ResourceApk resourceApk =
          ProcessedAndroidData.processBinaryDataFrom(
                  ctx,
                  errorReporter,
                  stampedManifest,
                  AndroidBinary.shouldShrinkResourceCycles(
                      ctx.getAndroidConfig(), errorReporter, settings.shrinkResources),
                  manifestValues,
                  settings.aaptVersion,
                  AndroidResources.from(
                      errorReporter, getFileProviders(resources), "resource_files"),
                  AndroidAssets.from(
                      errorReporter,
                      listFromNoneable(assets, ConfiguredTarget.class),
                      isNone(assetsDir)
                          ? null
                          : PathFragment.create(fromNoneable(assetsDir, String.class))),
                  resourceDeps,
                  AssetDependencies.fromProviders(
                      getProviders(deps, AndroidAssetsInfo.PROVIDER), /* neverlink = */ false),
                  settings.resourceFilterFactory,
                  settings.noCompressExtensions,
                  crunchPng,
                  /* featureOf = */ null,
                  /* featureAfter = */ null,
                  DataBinding.contextFrom(
                      dataBindingEnabled,
                      ctx.getActionConstructionContext(),
                      ctx.getAndroidConfig()))
              .generateRClass(ctx, settings.aaptVersion);

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
      SkylarkList<ConfiguredTarget> deps,
      SkylarkList<ConfiguredTarget> localProguardSpecs,
      SkylarkList<ConfiguredTarget> extraProguardSpecs,
      Location location,
      Environment env)
      throws EvalException, InterruptedException {
    BinaryDataSettings settings =
        fromNoneableOrDefault(
            maybeSettings, BinaryDataSettings.class, defaultBinaryDataSettings(ctx, location, env));

    if (!settings.shrinkResources) {
      return binaryDataInfo;
    }

    ImmutableList<Artifact> proguardSpecs =
        AndroidBinary.getProguardSpecs(
            ctx,
            getAndroidSemantics(),
            binaryDataInfo.getResourceProguardConfig(),
            binaryDataInfo.getManifestInfo().getManifest(),
            filesFromConfiguredTargets(localProguardSpecs),
            filesFromConfiguredTargets(extraProguardSpecs),
            getProviders(deps, ProguardSpecProvider.PROVIDER));

    // TODO(asteinb): There should never be more than one direct resource exposed in the provider.
    // Can we adjust its structure to take this into account?
    if (!binaryDataInfo.getResourcesInfo().getDirectAndroidResources().isSingleton()) {
      throw new EvalException(
          Location.BUILTIN,
          "Expected exactly 1 direct android resource container, but found: "
              + binaryDataInfo.getResourcesInfo().getDirectAndroidResources());
    }

    Optional<Artifact> maybeShrunkApk =
        AndroidBinary.maybeShrinkResources(
            ctx,
            binaryDataInfo.getResourcesInfo().getDirectAndroidResources().toList().get(0),
            ResourceDependencies.fromProviders(
                getProviders(deps, AndroidResourcesInfo.PROVIDER), /* neverlink = */ false),
            proguardSpecs,
            proguardOutputJar,
            proguardMapping,
            settings.aaptVersion,
            settings.resourceFilterFactory,
            settings.noCompressExtensions);

    return maybeShrunkApk.map(binaryDataInfo::withShrunkApk).orElse(binaryDataInfo);
  }

  public static SkylarkDict<Provider, NativeInfo> getNativeInfosFrom(
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

    return SkylarkDict.copyOf(/* env = */ null, builder.build());
  }

  /**
   * An algorithm to select the aapt version.
   *
   * <p>The calling rule doesn't have the aapt_version attribute (e.g. android_library), so fall
   * back to a simpler algorithm instead of {@code AndroidAaptVersion.chooseTargetAaptVersion}.
   * <li>1. If value of --android_aapt is either aapt or aapt2, use it.
   * <li>2. Else, use aapt2 if the sdk contains it. If it doesn't, use aapt.
   */
  private static AndroidAaptVersion getAndroidAaptVersionForLibrary(AndroidDataContext ctx) {
    AndroidAaptVersion aaptVersion = ctx.getAndroidConfig().getAndroidAaptVersion();
    if (aaptVersion == AndroidAaptVersion.AUTO) {
      aaptVersion =
          ctx.getSdk().getAapt2() == null ? AndroidAaptVersion.AAPT : AndroidAaptVersion.AAPT2;
    }
    return aaptVersion;
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
    return object == Runtime.NONE;
  }

  /**
   * Converts a "Noneable" Object passed by Skylark to an nullable object of the appropriate type.
   *
   * <p>Skylark "Noneable" types are passed in as an Object that may be either the correct type or a
   * Runtime.NONE object. Skylark will handle type checking, based on the appropriate @param
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
   * <p>This first calls {@link #fromNoneable(Object, Class)} to get a SkylarkList<?>, then safely
   * casts it to a list with the appropriate generic.
   */
  @Nullable
  public static <T> List<T> listFromNoneable(Object object, Class<T> clazz) throws EvalException {
    SkylarkList<?> asList = fromNoneable(object, SkylarkList.class);
    if (asList == null) {
      return null;
    }

    return SkylarkList.castList(asList, clazz, null);
  }

  private static ImmutableList<Artifact> filesFromConfiguredTargets(
      SkylarkList<ConfiguredTarget> targets) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (FileProvider provider : getFileProviders(targets)) {
      builder.addAll(provider.getFilesToBuild());
    }

    return builder.build();
  }

  private static ImmutableList<FileProvider> getFileProviders(
      SkylarkList<ConfiguredTarget> targets) {
    return getProviders(targets, FileProvider.class);
  }

  private static <T extends TransitiveInfoProvider> ImmutableList<T> getProviders(
      SkylarkList<ConfiguredTarget> targets, Class<T> clazz) {
    return targets
        .stream()
        .map(target -> target.getProvider(clazz))
        .filter(Objects::nonNull)
        .collect(ImmutableList.toImmutableList());
  }

  public static <T extends NativeInfo> SkylarkList<T> getProviders(
      SkylarkList<ConfiguredTarget> targets, NativeProvider<T> provider) {
    return SkylarkList.createImmutable(
        targets
            .stream()
            .map(target -> target.get(provider))
            .filter(Objects::nonNull)
            .collect(ImmutableList.toImmutableList()));
  }

  protected static <T extends NativeInfo> SkylarkList<T> getProviders(
      SkylarkList<ConfiguredTarget> targets, BuiltinProvider<T> provider) {
    return SkylarkList.createImmutable(
        targets
            .stream()
            .map(target -> target.get(provider))
            .filter(Objects::nonNull)
            .collect(ImmutableList.toImmutableList()));
  }
}
