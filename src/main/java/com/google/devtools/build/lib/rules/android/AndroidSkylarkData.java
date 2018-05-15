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
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarInfo.Aar;
import com.google.devtools.build.lib.rules.java.JavaCompilationInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.ProguardSpecProvider;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
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
@SkylarkModule(
    name = "android_data",
    doc =
        "Utilities for working with Android data (manifests, resources, and assets). "
            + "This API is non-final and subject to change without warning; do not rely on it.")
public abstract class AndroidSkylarkData {

  public abstract AndroidSemantics getAndroidSemantics();

  /**
   * Skylark API for getting a asset provider for android_library targets that don't specify assets.
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "assets_from_deps",
      mandatoryPositionals = 1, // context
      parameters = {
        @Param(
            name = "deps",
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = AndroidAssetsInfo.class,
            positional = false,
            named = true,
            doc = "Dependencies to inherit assets from"),
        @Param(
            name = "neverlink",
            defaultValue = "False",
            type = Boolean.class,
            positional = false,
            named = true,
            doc =
                "Defaults to False. If true, assets will not be exposed to targets that depend on"
                    + " them.")
      },
      doc =
          "Creates an AndroidAssetsInfo from this target's asset dependencies, ignoring local"
              + " assets. No processing will be done. This method is deprecated and exposed only"
              + " for backwards-compatibility with existing Native behavior.")
  public static AndroidAssetsInfo assetsFromDeps(
      SkylarkRuleContext ctx, SkylarkList<AndroidAssetsInfo> deps, boolean neverlink)
      throws EvalException {
    return AssetDependencies.fromProviders(deps, neverlink).toInfo(ctx.getLabel());
  }

  /**
   * Skylark API for getting a resource provider for android_library targets that don't specify
   * resources.
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "resources_from_deps",
      mandatoryPositionals = 1, // context
      parameters = {
        @Param(
            name = "deps",
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = AndroidResourcesInfo.class,
            positional = false,
            named = true,
            doc = "Dependencies to inherit resources from"),
        @Param(
            name = "neverlink",
            defaultValue = "False",
            type = Boolean.class,
            positional = false,
            named = true,
            doc =
                "Defaults to False. If true, resources will not be exposed to targets that depend"
                    + " on them."),
        @Param(
            name = "custom_package",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "The Android application package to stamp the manifest with. If not provided, the"
                    + " current Java package, derived from the location of this target's BUILD"
                    + " file, will be used. For example, given a BUILD file in"
                    + " 'java/com/foo/bar/BUILD', the package would be 'com.foo.bar'."),
      },
      doc =
          "Creates an AndroidResourcesInfo from this target's resource dependencies, ignoring local"
              + " resources. Only processing of deps will be done. This method is deprecated and"
              + " exposed only for backwards-compatibility with existing Native behavior. An empty"
              + " manifest will be generated and included in the provider - this path should not"
              + " be used when an explicit manifest is specified.")
  public static AndroidResourcesInfo resourcesFromDeps(
      SkylarkRuleContext ctx,
      SkylarkList<AndroidResourcesInfo> deps,
      boolean neverlink,
      Object customPackage)
      throws EvalException, InterruptedException {
    String pkg = fromNoneable(customPackage, String.class);
    if (pkg == null) {
      pkg = AndroidManifest.getDefaultPackage(ctx.getRuleContext());
    }
    return ResourceApk.processFromTransitiveLibraryData(
            ctx.getRuleContext(),
            ResourceDependencies.fromProviders(deps, /* neverlink = */ neverlink),
            AssetDependencies.empty(),
            StampedAndroidManifest.createEmpty(ctx.getRuleContext(), pkg, /* exported = */ false))
        .toResourceInfo(ctx.getLabel());
  }

  /**
   * Skylark API for stamping an Android manifest
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "stamp_manifest",
      mandatoryPositionals = 1, // SkylarkRuleContext ctx is mandatory
      parameters = {
        @Param(
            name = "manifest",
            positional = false,
            defaultValue = "None",
            type = Artifact.class,
            noneable = true,
            named = true,
            doc = "The manifest to stamp. If not passed, a dummy manifest will be generated"),
        @Param(
            name = "custom_package",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "The Android application package to stamp the manifest with. If not provided, the"
                    + " current Java package, derived from the location of this target's BUILD"
                    + " file, will be used. For example, given a BUILD file in"
                    + " 'java/com/foo/bar/BUILD', the package would be 'com.foo.bar'."),
        @Param(
            name = "exports_manifest",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If passed as True, this manifest will be exported to and"
                    + " eventually merged into targets that depend on it. Otherwise, it won't be"
                    + " inherited."),
      },
      doc = "Stamps a manifest with package information.")
  public AndroidManifestInfo stampAndroidManifest(
      SkylarkRuleContext ctx, Object manifest, Object customPackage, boolean exported)
      throws InterruptedException {
    String pkg = fromNoneable(customPackage, String.class);
    if (pkg == null) {
      pkg = AndroidManifest.getDefaultPackage(ctx.getRuleContext());
    }

    Artifact primaryManifest = fromNoneable(manifest, Artifact.class);
    if (primaryManifest == null) {
      return StampedAndroidManifest.createEmpty(ctx.getRuleContext(), pkg, exported).toProvider();
    }

    // If needed, rename the manifest to "AndroidManifest.xml", which aapt expects.
    Artifact renamedManifest =
        getAndroidSemantics().renameManifest(ctx.getRuleContext(), primaryManifest);

    return new AndroidManifest(renamedManifest, pkg, exported)
        .stamp(ctx.getRuleContext())
        .toProvider();
  }

  /**
   * Skylark API for merging android_library assets
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "merge_assets",
      mandatoryPositionals = 1, // context
      parameters = {
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            noneable = true,
            named = true,
            doc =
                "Targets containing raw assets for this target. If passed, 'assets_dir' must also"
                    + " be passed."),
        @Param(
            name = "assets_dir",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "Directory the assets are contained in. Must be passed if and only if 'assets' is"
                    + " passed. This path will be split off of the asset paths on the device."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = AndroidAssetsInfo.class,
            named = true,
            doc =
                "Providers containing assets from dependencies. These assets will be merged"
                    + " together with each other and this target's assets."),
        @Param(
            name = "neverlink",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If passed as True, these assets will not be inherited by"
                    + " targets that depend on this one.")
      },
      doc =
          "Merges this target's assets together with assets inherited from dependencies. Note that,"
              + " by default, actions for validating the merge are created but may not be called."
              + " You may want to force these actions to be called - see the 'validation_result'"
              + " field in AndroidAssetsInfo")
  public AndroidAssetsInfo mergeAssets(
      SkylarkRuleContext ctx,
      Object assets,
      Object assetsDir,
      SkylarkList<AndroidAssetsInfo> deps,
      boolean neverlink)
      throws EvalException, InterruptedException {
    try {
      return AndroidAssets.from(
              ctx.getRuleContext(),
              listFromNoneable(assets, ConfiguredTarget.class),
              isNone(assetsDir) ? null : PathFragment.create(fromNoneable(assetsDir, String.class)))
          .parse(ctx.getRuleContext())
          .merge(
              ctx.getRuleContext(),
              AssetDependencies.fromProviders(deps.getImmutableList(), neverlink))
          .toProvider();
    } catch (RuleErrorException e) {
      throw new EvalException(Location.BUILTIN, e);
    }
  }

  /**
   * Skylark API for merging android_library resources
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "merge_resources",
      mandatoryPositionals = 2, // context, manifest
      parameters = {
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = FileProvider.class,
            named = true,
            doc = "Providers of this target's resources"),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = AndroidResourcesInfo.class,
            named = true,
            doc =
                "Targets containing raw resources from dependencies. These resources will be merged"
                    + " together with each other and this target's resources."),
        @Param(
            name = "neverlink",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If passed as True, these resources will not be inherited by"
                    + " targets that depend on this one."),
        @Param(
            name = "enable_data_binding",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If True, processes data binding expressions in layout"
                    + " resources."),
      },
      doc =
          "Merges this target's resources together with resources inherited from dependencies."
              + " Returns a dict of provider type to actual info, with elements for"
              + " AndroidResourcesInfo (various resource information) and JavaInfo (wrapping the"
              + " R.class jar, for use in Java compilation). The passed manifest provider is used"
              + " to get Android package information and to validate that all resources it refers"
              + " to are available. Note that this method might do additional processing to this"
              + " manifest, so in the future, you may want to use the manifest contained in this"
              + " method's output instead of this one.")
  public SkylarkDict<NativeProvider<?>, NativeInfo> mergeResources(
      SkylarkRuleContext ctx,
      AndroidManifestInfo manifest,
      SkylarkList<ConfiguredTarget> resources,
      SkylarkList<AndroidResourcesInfo> deps,
      boolean neverlink,
      boolean enableDataBinding)
      throws EvalException, InterruptedException {
    try {
      AndroidAaptVersion aaptVersion =
          AndroidCommon.getAndroidConfig(ctx.getRuleContext()).getAndroidAaptVersion();

      ValidatedAndroidResources validated =
          AndroidResources.from(ctx.getRuleContext(), getFileProviders(resources), "resources")
              .process(
                  ctx.getRuleContext(),
                  manifest.asStampedManifest(),
                  ResourceDependencies.fromProviders(deps, neverlink),
                  enableDataBinding,
                  aaptVersion)
              .validate(ctx.getRuleContext(), aaptVersion);

      JavaInfo javaInfo = getJavaInfoForRClassJar(validated.getClassJar());

      return SkylarkDict.of(
          /* env = */ null,
          AndroidResourcesInfo.PROVIDER,
          validated.toProvider(),
          JavaInfo.PROVIDER,
          javaInfo);

    } catch (RuleErrorException e) {
      throw new EvalException(Location.BUILTIN, e);
    }
  }

  /**
   * Skylark API for building an Aar for an android_library
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "make_aar",
      mandatoryPositionals = 4, // context, resource info, asset info, and library class jar
      parameters = {
        @Param(
            name = "proguard_specs",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            defaultValue = "[]",
            positional = false,
            named = true,
            doc =
                "Files to be used as Proguard specification for this target, which will be"
                    + " inherited in the top-level target"),
        @Param(
            name = "deps",
            type = SkylarkList.class,
            generic1 = AndroidLibraryAarInfo.class,
            defaultValue = "[]",
            positional = false,
            named = true,
            doc = "Dependant AAR providers used to build this AAR."),
        @Param(
            name = "neverlink",
            type = Boolean.class,
            defaultValue = "False",
            positional = false,
            named = true,
            doc =
                "Defaults to False. If true, this target's Aar will not be generated or propagated"
                    + " to targets that depend upon it."),
      },
      doc =
          "Builds an AAR and corresponding provider for this target. The resource and asset"
              + " providers from this same target must both be passed, as must the class JAR output"
              + " of building the Android Java library.")
  public AndroidLibraryAarInfo makeAar(
      SkylarkRuleContext ctx,
      AndroidResourcesInfo resourcesInfo,
      AndroidAssetsInfo assetsInfo,
      Artifact libraryClassJar,
      SkylarkList<ConfiguredTarget> proguardSpecs,
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
      ValidatedAndroidData validatedAndroidData =
          resourcesInfo.getDirectAndroidResources().toList().get(0);
      if (validatedAndroidData.getLabel().equals(ctx.getLabel())) {
        // TODO(b/77574966): Remove this cast once we get rid of ResourceContainer and can guarantee
        // that only properly processed resources are passed into this object.
        if (!(validatedAndroidData instanceof ValidatedAndroidResources)) {
          throw new EvalException(
              Location.BUILTIN, "Old data processing pipeline does not support the Skylark API");
        }
        resources = (ValidatedAndroidResources) validatedAndroidData;
      } else {
        definesLocalResources = false;
      }
    }

    // Get the target's local assets, if defined, from the provider
    boolean definesLocalAssets = assetsInfo.getDirectParsedAssets().isSingleton();
    AndroidAssets assets = AndroidAssets.empty();
    if (definesLocalAssets) {
      ParsedAndroidAssets parsed = assetsInfo.getDirectParsedAssets().toList().get(0);
      if (parsed.getLabel().equals(ctx.getLabel())) {
        assets = parsed;
      } else {
        definesLocalAssets = false;
      }
    }

    if (definesLocalResources != definesLocalAssets) {
      throw new EvalException(
          Location.BUILTIN,
          "Must define either both or none of assets and resources. Use the merge_assets and"
              + " merge_resources methods to define them, or assets_from_deps and"
              + " resources_from_deps to inherit without defining them.");
    }

    return Aar.makeAar(
            ctx.getRuleContext(),
            resources,
            assets,
            resourcesInfo.getManifest(),
            resourcesInfo.getRTxt(),
            libraryClassJar,
            filesFromConfiguredTargets(proguardSpecs))
        .toProvider(deps, definesLocalResources);
  }

  /**
   * Skylark API for doing all resource, asset, and manifest processing for an android_library
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "process_library_data",
      mandatoryPositionals = 2, // ctx and libraryClassJar are required
      parameters = {
        @Param(
            name = "manifest",
            positional = false,
            type = Artifact.class,
            defaultValue = "None",
            named = true,
            noneable = true,
            doc =
                "If passed, the manifest to use for this target. Otherwise, a dummy manifest will"
                    + " be generated."),
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "None",
            type = SkylarkList.class,
            generic1 = FileProvider.class,
            named = true,
            noneable = true,
            doc = "Providers of this target's resources"),
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            noneable = true,
            named = true,
            doc =
                "Targets containing raw assets for this target. If passed, 'assets_dir' must also"
                    + " be passed."),
        @Param(
            name = "assets_dir",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "Directory the assets are contained in. Must be passed if and only if 'assets' is"
                    + " passed. This path will be split off of the asset paths on the device."),
        @Param(
            name = "exports_manifest",
            positional = false,
            defaultValue = "None",
            type = Boolean.class,
            named = true,
            noneable = true,
            doc =
                "Defaults to False. If passed as True, this manifest will be exported to and"
                    + " eventually merged into targets that depend on it. Otherwise, it won't be"
                    + " inherited."),
        @Param(
            name = "custom_package",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "The Android application package to stamp the manifest with. If not provided, the"
                    + " current Java package, derived from the location of this target's BUILD"
                    + " file, will be used. For example, given a BUILD file in"
                    + " 'java/com/foo/bar/BUILD', the package would be 'com.foo.bar'."),
        @Param(
            name = "neverlink",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If passed as True, these resources and assets will not be"
                    + " inherited by targets that depend on this one."),
        @Param(
            name = "enable_data_binding",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If True, processes data binding expressions in layout"
                    + " resources."),
        @Param(
            name = "proguard_specs",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            defaultValue = "[]",
            positional = false,
            named = true,
            doc =
                "Files to be used as Proguard specification for this target, which will be"
                    + " inherited in the top-level target"),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = AndroidAssetsInfo.class,
            named = true,
            doc =
                "Dependency targets. Providers will be extracted from these dependencies for each"
                    + " type of data."),
      },
      doc =
          "Performs full processing of data for android_library or similar rules. Returns a dict"
              + " from provider type to providers for the target.")
  public SkylarkDict<NativeProvider<?>, NativeInfo> processLibraryData(
      SkylarkRuleContext ctx,
      Artifact libraryClassJar,
      Object manifest,
      Object resources,
      Object assets,
      Object assetsDir,
      Object exportsManifest,
      Object customPackage,
      boolean neverlink,
      boolean enableDataBinding,
      SkylarkList<ConfiguredTarget> proguardSpecs,
      SkylarkList<ConfiguredTarget> deps)
      throws InterruptedException, EvalException {

    SkylarkList<AndroidResourcesInfo> resourceDeps =
        getProviders(deps, AndroidResourcesInfo.PROVIDER);
    SkylarkList<AndroidAssetsInfo> assetDeps = getProviders(deps, AndroidAssetsInfo.PROVIDER);

    ImmutableMap.Builder<NativeProvider<?>, NativeInfo> infoBuilder = ImmutableMap.builder();

    AndroidResourcesInfo resourcesInfo;
    AndroidAssetsInfo assetsInfo;
    if (isNone(manifest)
        && isNone(resources)
        && isNone(assets)
        && isNone(assetsDir)
        && isNone(exportsManifest)) {

      // If none of these parameters were specified, for backwards compatibility, do not trigger
      // data processing.
      resourcesInfo = resourcesFromDeps(ctx, resourceDeps, neverlink, customPackage);
      assetsInfo = assetsFromDeps(ctx, assetDeps, neverlink);

      infoBuilder.put(AndroidResourcesInfo.PROVIDER, resourcesInfo);
    } else {

      AndroidManifestInfo baseManifest =
          stampAndroidManifest(
              ctx,
              manifest,
              customPackage,
              fromNoneableOrDefault(exportsManifest, Boolean.class, false));

      SkylarkDict<NativeProvider<?>, NativeInfo> resourceOutput =
          mergeResources(
              ctx,
              baseManifest,
              listFromNoneableOrEmpty(resources, ConfiguredTarget.class),
              resourceDeps,
              neverlink,
              enableDataBinding);

      resourcesInfo = (AndroidResourcesInfo) resourceOutput.get(AndroidResourcesInfo.PROVIDER);
      assetsInfo = mergeAssets(ctx, assets, assetsDir, assetDeps, neverlink);

      infoBuilder.putAll(resourceOutput);
    }


    AndroidLibraryAarInfo aarInfo =
        makeAar(
            ctx,
            resourcesInfo,
            assetsInfo,
            libraryClassJar,
            proguardSpecs,
            getProviders(deps, AndroidLibraryAarInfo.PROVIDER),
            neverlink);

    // Only expose the aar provider in non-neverlinked actions
    if (!neverlink) {
      infoBuilder.put(AndroidLibraryAarInfo.PROVIDER, aarInfo);
    }

    // Expose the updated manifest that was changed by resource processing
    // TODO(b/30817309): Use the base manifest once manifests are no longer changed in resource
    // processing
    AndroidManifestInfo manifestInfo = resourcesInfo.getManifest().toProvider();

    return SkylarkDict.copyOf(
        /* env = */ null,
        infoBuilder
            .put(AndroidAssetsInfo.PROVIDER, assetsInfo)
            .put(AndroidManifestInfo.PROVIDER, manifestInfo)
            .build());
  }

  /**
   * Skylark API for doing all resource, asset, and manifest processing for an aar_import target
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "process_aar_import_data",
      // context, resource and asset TreeArtifacts, and manifest artifact are all mandatory
      mandatoryPositionals = 4,
      parameters = {
        @Param(
            name = "deps",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Targets to inherit asset and resource dependencies from.")
      },
      doc = "Processes assets, resources, and manifest for aar_import targets")
  public SkylarkDict<NativeProvider<?>, NativeInfo> processAarImportData(
      SkylarkRuleContext ctx,
      SpecialArtifact resources,
      SpecialArtifact assets,
      Artifact androidManifestArtifact,
      SkylarkList<ConfiguredTarget> deps)
      throws EvalException, InterruptedException {

    AndroidAaptVersion aaptVersion =
        AndroidCommon.getAndroidConfig(ctx.getRuleContext()).getAndroidAaptVersion();

    ValidatedAndroidResources validatedResources =
        AndroidResources.forAarImport(resources)
            .process(
                ctx.getRuleContext(),
                AndroidManifest.forAarImport(androidManifestArtifact),
                ResourceDependencies.fromProviders(
                    getProviders(deps, AndroidResourcesInfo.PROVIDER), /* neverlink = */ false),
                /* enableDataBinding = */ false,
                aaptVersion);

    MergedAndroidAssets mergedAssets =
        AndroidAssets.forAarImport(assets)
            .parse(ctx.getRuleContext())
            .merge(
                ctx.getRuleContext(),
                AssetDependencies.fromProviders(
                    getProviders(deps, AndroidAssetsInfo.PROVIDER), /* neverlink = */ false));

    ResourceApk resourceApk = ResourceApk.of(validatedResources, mergedAssets, null, null);

    return getNativeInfosFrom(resourceApk, ctx.getLabel());
  }

  /**
   * Skylark API for processing assets, resources, and manifest for android_local_test
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "process_local_test_data",
      mandatoryPositionals = 1, // context is mandatory
      parameters = {
        @Param(
            name = "manifest",
            positional = false,
            type = Artifact.class,
            defaultValue = "None",
            named = true,
            noneable = true,
            doc =
                "If passed, the manifest to use for this target. Otherwise, a dummy manifest will"
                    + " be generated."),
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = FileProvider.class,
            named = true,
            doc = "Providers of this target's resources"),
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            noneable = true,
            named = true,
            doc =
                "Targets containing raw assets for this target. If passed, 'assets_dir' must also"
                    + " be passed."),
        @Param(
            name = "assets_dir",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "Directory the assets are contained in. Must be passed if and only if 'assets' is"
                    + " passed. This path will be split off of the asset paths on the device."),
        @Param(
            name = "custom_package",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "The Android application package to stamp the manifest with. If not provided, the"
                    + " current Java package, derived from the location of this target's BUILD"
                    + " file, will be used. For example, given a BUILD file in"
                    + " 'java/com/foo/bar/BUILD', the package would be 'com.foo.bar'."),
        @Param(
            name = "aapt_version",
            positional = false,
            defaultValue = "'auto'",
            type = String.class,
            named = true,
            doc =
                "The version of aapt to use. Defaults to 'auto'. 'aapt' and 'aapt2' are also"
                    + " supported."),
        @Param(
            name = "manifest_values",
            positional = false,
            defaultValue = "{}",
            type = SkylarkDict.class,
            generic1 = String.class,
            named = true,
            doc = "A dictionary of values to be overridden in the manifest."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = AndroidAssetsInfo.class,
            named = true,
            doc =
                "Dependency targets. Providers will be extracted from these dependencies for each"
                    + " type of data."),
      },
      doc =
          "Processes resources, assets, and manifests for android_local_test and returns a dict"
              + " from provider type to the appropriate provider.")
  public SkylarkDict<NativeProvider<?>, NativeInfo> processLocalTestData(
      SkylarkRuleContext ctx,
      Object manifest,
      SkylarkList<ConfiguredTarget> resources,
      Object assets,
      Object assetsDir,
      Object customPackage,
      String aaptVersionString,
      SkylarkDict<String, String> rawManifestValues,
      SkylarkList<ConfiguredTarget> deps)
      throws InterruptedException, EvalException {

    AndroidManifest rawManifest =
        AndroidManifest.from(
            ctx.getRuleContext(),
            fromNoneable(manifest, Artifact.class),
            fromNoneable(customPackage, String.class),
            /* exportsManifest = */ false);

    try {
      ResourceApk resourceApk =
          AndroidLocalTestBase.buildResourceApk(
              ctx.getRuleContext(),
              rawManifest,
              AndroidResources.from(
                  ctx.getRuleContext(), getFileProviders(resources), "resource_files"),
              AndroidAssets.from(
                  ctx.getRuleContext(),
                  listFromNoneable(assets, ConfiguredTarget.class),
                  isNone(assetsDir)
                      ? null
                      : PathFragment.create(fromNoneable(assetsDir, String.class))),
              ResourceDependencies.fromProviders(
                  getProviders(deps, AndroidResourcesInfo.PROVIDER), /* neverlink = */ false),
              AssetDependencies.fromProviders(
                  getProviders(deps, AndroidAssetsInfo.PROVIDER), /* neverlink = */ false),
              ApplicationManifest.getManifestValues(ctx.getRuleContext(), rawManifestValues),
              AndroidAaptVersion.chooseTargetAaptVersion(
                  ctx.getRuleContext(),
                  AndroidCommon.getAndroidConfig(ctx.getRuleContext()),
                  aaptVersionString));

      return getNativeInfosFrom(resourceApk, ctx.getLabel());
    } catch (RuleErrorException e) {
      throw new EvalException(Location.BUILTIN, e);
    }
  }

  /**
   * Skylark API for bundling common setting for working with resources in android_binary
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "make_binary_settings",
      mandatoryPositionals = 1, // SkylarkRuleContext is mandatory
      parameters = {
        @Param(
            name = "shrink_resources",
            positional = false,
            noneable = true,
            defaultValue = "None",
            type = Boolean.class,
            named = true,
            doc =
                "Whether to shrink resources. Defaults to the value used in Android"
                    + " configuration."),
        @Param(
            name = "resource_configuration_filters",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = String.class,
            named = true,
            doc =
                "A list of resource configuration filters, such 'en' that will limit the resources"
                    + " in the apk to only the ones in the 'en' configuration."),
        @Param(
            name = "densities",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = String.class,
            named = true,
            doc =
                "Densities to filter for when building the apk. A corresponding compatible-screens"
                    + " section will also be added to the manifest if it does not already contain a"
                    + " superset listing."),
        @Param(
            name = "nocompress_extensions",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = String.class,
            named = true,
            doc = "A list of file extension to leave uncompressed in apk."),
        @Param(
            name = "aapt_version",
            positional = false,
            defaultValue = "'auto'",
            type = String.class,
            named = true,
            doc =
                "The version of aapt to use. Defaults to 'auto'. 'aapt' and 'aapt2' are also"
                    + " supported."),
      },
      doc =
          "Returns a wrapper object containing various settings shared across multiple methods for"
              + " processing binary data.")
  public BinaryDataSettings makeBinarySettings(
      SkylarkRuleContext ctx,
      Object shrinkResources,
      SkylarkList<String> resourceConfigurationFilters,
      SkylarkList<String> densities,
      SkylarkList<String> rawNoCompressExtensions,
      String aaptVersionString)
      throws EvalException {
    AndroidConfiguration androidConfig = AndroidCommon.getAndroidConfig(ctx.getRuleContext());

    AndroidAaptVersion aaptVersion;
    try {
      aaptVersion =
          AndroidAaptVersion.chooseTargetAaptVersion(
              ctx.getRuleContext(), androidConfig, aaptVersionString);
    } catch (RuleErrorException e) {
      throw new EvalException(Location.BUILTIN, e);
    }

    return new BinaryDataSettings(
        aaptVersion,
        fromNoneableOrDefault(
            shrinkResources, Boolean.class, androidConfig.useAndroidResourceShrinking()),
        ResourceFilterFactory.from(aaptVersion, resourceConfigurationFilters, densities),
        ctx.getRuleContext()
            .getExpander()
            .withDataLocations()
            .tokenized("nocompress_extensions", rawNoCompressExtensions));
  }

  /**
   * Helper method to get default {@link
   * com.google.devtools.build.lib.rules.android.AndroidSkylarkData.BinaryDataSettings}.
   */
  private BinaryDataSettings defaultBinaryDataSettings(SkylarkRuleContext ctx)
      throws EvalException {
    return makeBinarySettings(
        ctx,
        Runtime.NONE,
        SkylarkList.createImmutable(ImmutableList.of()),
        SkylarkList.createImmutable(ImmutableList.of()),
        SkylarkList.createImmutable(ImmutableList.of()),
        "auto");
  }

  @SkylarkModule(
      name = "AndroidBinaryDataSettings",
      doc = "Wraps common settings for working with android_binary assets, resources, and manifest")
  private static class BinaryDataSettings {
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

  /**
   * Skylark API for processing assets, resources, and manifest for android_binary
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "process_binary_data",
      mandatoryPositionals = 1, // SkylarkRuleContext is mandatory
      parameters = {
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = FileProvider.class,
            named = true,
            doc = "Providers of this target's resources"),
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            noneable = true,
            named = true,
            doc =
                "Targets containing raw assets for this target. If passed, 'assets_dir' must also"
                    + " be passed."),
        @Param(
            name = "assets_dir",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "Directory the assets are contained in. Must be passed if and only if 'assets' is"
                    + " passed. This path will be split off of the asset paths on the device."),
        @Param(
            name = "manifest",
            positional = false,
            type = Artifact.class,
            defaultValue = "None",
            named = true,
            noneable = true,
            doc =
                "If passed, the manifest to use for this target. Otherwise, a dummy manifest will"
                    + " be generated."),
        @Param(
            name = "custom_package",
            positional = false,
            defaultValue = "None",
            type = String.class,
            noneable = true,
            named = true,
            doc =
                "The Android application package to stamp the manifest with. If not provided, the"
                    + " current Java package, derived from the location of this target's BUILD"
                    + " file, will be used. For example, given a BUILD file in"
                    + " 'java/com/foo/bar/BUILD', the package would be 'com.foo.bar'."),
        @Param(
            name = "manifest_values",
            positional = false,
            defaultValue = "{}",
            type = SkylarkDict.class,
            generic1 = String.class,
            named = true,
            doc = "A dictionary of values to be overridden in the manifest."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            named = true,
            doc =
                "Dependency targets. Providers will be extracted from these dependencies for each"
                    + " type of data."),
        @Param(
            name = "manifest_merger",
            type = String.class,
            defaultValue = "'auto'",
            positional = false,
            named = true,
            doc =
                "The manifest merger to use. Defaults to 'auto', but 'android' and 'legacy' are"
                    + " also supported."),
        @Param(
            name = "binary_settings",
            type = BinaryDataSettings.class,
            noneable = true,
            defaultValue = "None",
            positional = false,
            named = true,
            doc =
                "Settings common to various binary processing methods, created with"
                    + " make_binary_data_settings"),
        @Param(
            name = "crunch_png",
            positional = false,
            defaultValue = "True",
            type = Boolean.class,
            named = true,
            doc = "Whether PNG crunching should be done. Defaults to True."),
        @Param(
            name = "enable_data_binding",
            positional = false,
            defaultValue = "False",
            type = Boolean.class,
            named = true,
            doc =
                "Defaults to False. If True, processes data binding expressions in layout"
                    + " resources."),
      },
      doc =
          "Processes resources, assets, and manifests for android_binary and returns the"
              + " appropriate providers.")
  public AndroidBinaryDataInfo processBinaryData(
      SkylarkRuleContext ctx,
      SkylarkList<ConfiguredTarget> resources,
      Object assets,
      Object assetsDir,
      Object manifest,
      Object customPackage,
      SkylarkDict<String, String> rawManifestValues,
      SkylarkList<ConfiguredTarget> deps,
      String manifestMerger,
      Object maybeSettings,
      boolean crunchPng,
      boolean dataBindingEnabled)
      throws InterruptedException, RuleErrorException, EvalException {

    BinaryDataSettings settings =
        fromNoneableOrDefault(
            maybeSettings, BinaryDataSettings.class, defaultBinaryDataSettings(ctx));

    AndroidManifest rawManifest =
        AndroidManifest.from(
            ctx.getRuleContext(),
            fromNoneable(manifest, Artifact.class),
            getAndroidSemantics(),
            fromNoneable(customPackage, String.class),
            /* exportsManifest = */ false);

    ResourceDependencies resourceDeps =
        ResourceDependencies.fromProviders(
            getProviders(deps, AndroidResourcesInfo.PROVIDER), /* neverlink = */ false);

    ImmutableMap<String, String> manifestValues =
        ApplicationManifest.getManifestValues(ctx.getRuleContext(), rawManifestValues);

    StampedAndroidManifest stampedManifest =
        rawManifest.mergeWithDeps(
            ctx.getRuleContext(),
            resourceDeps,
            manifestValues,
            ApplicationManifest.useLegacyMerging(ctx.getRuleContext(), manifestMerger));

    ResourceApk resourceApk =
        ProcessedAndroidData.processBinaryDataFrom(
                ctx.getRuleContext(),
                stampedManifest,
                AndroidBinary.shouldShrinkResourceCycles(
                    ctx.getRuleContext(), settings.shrinkResources),
                manifestValues,
                settings.aaptVersion,
                AndroidResources.from(
                    ctx.getRuleContext(), getFileProviders(resources), "resource_files"),
                AndroidAssets.from(
                    ctx.getRuleContext(),
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
                dataBindingEnabled)
            .generateRClass(ctx.getRuleContext(), settings.aaptVersion);

    return AndroidBinaryDataInfo.of(
        resourceApk.getArtifact(),
        resourceApk.getResourceProguardConfig(),
        resourceApk.toResourceInfo(ctx.getLabel()),
        resourceApk.toAssetsInfo(ctx.getLabel()).get(),
        resourceApk.toManifestInfo().get());
  }

  /**
   * Skylark API for shrinking a resource APK
   *
   * <p>TODO(b/79159379): Stop passing SkylarkRuleContext here
   *
   * @param ctx the SkylarkRuleContext. We will soon change to using an ActionConstructionContext
   *     instead. See b/79159379
   */
  @SkylarkCallable(
      name = "shrink_data_apk",
      // Required: SkylarkRuleContext, AndroidBinaryDataInfo to shrink, and two proguard outputs
      mandatoryPositionals = 4,
      parameters = {
        @Param(
            name = "binary_settings",
            type = BinaryDataSettings.class,
            noneable = true,
            defaultValue = "None",
            positional = false,
            named = true,
            doc =
                "Settings common to various binary processing methods, created with"
                    + " make_binary_data_settings"),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            named = true,
            doc =
                "Dependency targets. Providers will be extracted from these dependencies for each"
                    + " type of data."),
        @Param(
            name = "proguard_specs",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            defaultValue = "[]",
            positional = false,
            named = true,
            doc =
                "Files to be used as Proguard specification for this target, which will be"
                    + " inherited in the top-level target"),
        @Param(
            name = "extra_proguard_specs,",
            type = SkylarkList.class,
            generic1 = ConfiguredTarget.class,
            defaultValue = "[]",
            positional = false,
            named = true,
            doc =
                "Additional proguard specs that should be added for top-level targets. This  value"
                    + " is controlled by Java configuration."),
      },
      doc =
          "Possibly shrinks the data APK by removing resources that were marked as unused during"
              + " proguarding.")
  public AndroidBinaryDataInfo shrinkDataApk(
      SkylarkRuleContext ctx,
      AndroidBinaryDataInfo binaryDataInfo,
      Artifact proguardOutputJar,
      Artifact proguardMapping,
      Object maybeSettings,
      SkylarkList<ConfiguredTarget> deps,
      SkylarkList<ConfiguredTarget> localProguardSpecs,
      SkylarkList<ConfiguredTarget> extraProguardSpecs)
      throws EvalException, InterruptedException {
    BinaryDataSettings settings =
        fromNoneableOrDefault(
            maybeSettings, BinaryDataSettings.class, defaultBinaryDataSettings(ctx));

    if (!settings.shrinkResources) {
      return binaryDataInfo;
    }

    ImmutableList<Artifact> proguardSpecs =
        AndroidBinary.getProguardSpecs(
            ctx.getRuleContext(),
            getAndroidSemantics(),
            binaryDataInfo.getResourceProguardConfig(),
            binaryDataInfo.getManifestInfo().getManifest(),
            filesFromConfiguredTargets(localProguardSpecs),
            filesFromConfiguredTargets(extraProguardSpecs),
            getProviders(deps, ProguardSpecProvider.class));

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
            ctx.getRuleContext(),
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

  public static SkylarkDict<NativeProvider<?>, NativeInfo> getNativeInfosFrom(
      ResourceApk resourceApk, Label label) {
    ImmutableMap.Builder<NativeProvider<?>, NativeInfo> builder = ImmutableMap.builder();

    builder.put(AndroidResourcesInfo.PROVIDER, resourceApk.toResourceInfo(label));

    resourceApk
        .toAssetsInfo(label)
        .ifPresent(info -> builder.put(AndroidAssetsInfo.PROVIDER, info));
    resourceApk.toManifestInfo().ifPresent(info -> builder.put(AndroidManifestInfo.PROVIDER, info));

    builder.put(JavaInfo.PROVIDER, getJavaInfoForRClassJar(resourceApk.getResourceJavaClassJar()));

    return SkylarkDict.copyOf(/* env = */ null, builder.build());
  }

  private static JavaInfo getJavaInfoForRClassJar(Artifact rClassJar) {
    return JavaInfo.Builder.create()
        .setNeverlink(true)
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

  private static <T> SkylarkList<T> listFromNoneableOrEmpty(Object object, Class<T> clazz)
      throws EvalException {
    List<T> value = listFromNoneable(object, clazz);
    if (value == null) {
      return SkylarkList.createImmutable(ImmutableList.of());
    }

    return SkylarkList.createImmutable(value);
  }
}
