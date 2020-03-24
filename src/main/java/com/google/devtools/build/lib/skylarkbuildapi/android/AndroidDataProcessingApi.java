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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or Tied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.skylarkbuildapi.android;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.FileProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Skylark-visible methods for working with Android data (manifests, resources, and assets). */
@SkylarkModule(
    name = "android_data",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Utilities for working with Android data (manifests, resources, and assets). "
            + "This API is non-final and subject to change without warning; do not rely on it.",
    documented = false)
public interface AndroidDataProcessingApi<
        AndroidDataContextT extends AndroidDataContextApi,
        TransitiveInfoCollectionT extends TransitiveInfoCollectionApi,
        FileT extends FileApi,
        SpecialFileT extends FileApi,
        AndroidAssetsInfoT extends AndroidAssetsInfoApi<?, ?>,
        AndroidResourcesInfoT extends AndroidResourcesInfoApi<?, ?, ?>,
        AndroidManifestInfoT extends AndroidManifestInfoApi<?>,
        AndroidLibraryAarInfoT extends AndroidLibraryAarInfoApi<?>,
        AndroidBinaryDataInfoT extends AndroidBinaryDataInfoApi<?>,
        ValidatedAndroidDataT extends ValidatedAndroidDataApi<?, ?>>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "assets_from_deps",
      parameters = {
        @Param(
            name = "deps",
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = AndroidAssetsInfoApi.class,
            positional = false,
            named = true,
            doc = "Dependencies to inherit assets from."),
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
      useStarlarkThread = true,
      doc =
          "Creates an AndroidAssetsInfoApi from this target's asset dependencies, ignoring local"
              + " assets. No processing will be done. This method is deprecated and exposed only"
              + " for backwards-compatibility with existing behavior.",
      documented = false)
  AndroidAssetsInfoT assetsFromDeps(
      Sequence<?> deps, // <AndroidAssetsInfoT>
      boolean neverlink,
      StarlarkThread thread)
      throws EvalException;

  @SkylarkCallable(
      name = "resources_from_deps",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "deps",
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = AndroidResourcesInfoApi.class,
            positional = false,
            named = true,
            doc = "Dependencies to inherit resources from."),
        @Param(
            name = "assets",
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = AndroidAssetsInfoApi.class,
            positional = false,
            named = true,
            doc = "Dependencies to inherit assets from."),
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
            type = String.class,
            noneable = false,
            named = true,
            doc = "The Android application package to stamp the manifest with."),
      },
      doc =
          "Creates an AndroidResourcesInfoApi from this target's resource dependencies, ignoring"
              + " local resources. Only processing of deps will be done. This method is deprecated"
              + " and exposed only for backwards-compatibility with existing behavior. An empty"
              + " manifest will be generated and included in the provider - this path should  not"
              + " be used when an explicit manifest is specified.",
      documented = false)
  AndroidResourcesInfoT resourcesFromDeps(
      AndroidDataContextT ctx,
      Sequence<?> deps, // <AndroidResourcesInfoT>
      Sequence<?> assets, // <AndroidAssetsInfoT>
      boolean neverlink,
      String customPackage)
      throws InterruptedException, EvalException;

  @SkylarkCallable(
      name = "stamp_manifest",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "manifest",
            positional = false,
            defaultValue = "None",
            type = FileApi.class,
            noneable = true,
            named = true,
            doc = "The manifest to stamp. If not passed, a dummy manifest will be generated."),
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
      doc = "Stamps a manifest with package information.",
      documented = false)
  AndroidManifestInfoT stampAndroidManifest(
      AndroidDataContextT ctx, Object manifest, Object customPackage, boolean exported)
      throws InterruptedException, EvalException;

  @SkylarkCallable(
      name = "merge_assets",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
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
            type = Sequence.class,
            generic1 = AndroidAssetsInfoApi.class,
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
              + " field in AndroidAssetsInfoApi",
      documented = false)
  AndroidAssetsInfoT mergeAssets(
      AndroidDataContextT ctx,
      Object assets,
      Object assetsDir,
      Sequence<?> deps, // <AndroidAssetsInfoT>
      boolean neverlink)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "merge_res",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "manifest",
            positional = true,
            named = false,
            type = AndroidManifestInfoApi.class,
            doc =
                "The provider of this target's manifest. This provider is produced by, "
                    + "for example, stamp_android_manifest."),
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = FileProviderApi.class,
            named = true,
            doc = "Providers of this target's resources."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = AndroidResourcesInfoApi.class,
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
              + " AndroidResourcesInfoApi (various resource information) and JavaInfoApi (wrapping"
              + " the R.class jar, for use in Java compilation). The passed manifest provider is"
              + " used to get Android package information and to validate that all resources it"
              + " refers to are available. Note that this method might do additional processing to"
              + " this manifest, so in the future, you may want to use the manifest contained in"
              + " this method's output instead of this one.",
      documented = false,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  ValidatedAndroidDataT mergeRes(
      AndroidDataContextT ctx,
      AndroidManifestInfoT manifest,
      Sequence<?> resources, // <TransitiveInfoCollectionT>
      Sequence<?> deps, // <AndroidResourcesInfoT>
      boolean neverlink,
      boolean enableDataBinding)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "merge_resources",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "manifest",
            positional = true,
            named = false,
            type = AndroidManifestInfoApi.class,
            doc =
                "The provider of this target's manifest. This provider is produced by, "
                    + "for example, stamp_android_manifest."),
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = FileProviderApi.class,
            named = true,
            doc = "Providers of this target's resources."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = AndroidResourcesInfoApi.class,
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
              + " AndroidResourcesInfoApi (various resource information) and JavaInfoApi (wrapping"
              + " the R.class jar, for use in Java compilation). The passed manifest provider is"
              + " used to get Android package information and to validate that all resources it"
              + " refers to are available. Note that this method might do additional processing to"
              + " this manifest, so in the future, you may want to use the manifest contained in"
              + " this method's output instead of this one.",
      documented = false)
  Dict<? extends ProviderApi, ? extends StructApi> mergeResources(
      AndroidDataContextT ctx,
      AndroidManifestInfoT manifest,
      Sequence<?> resources, // <TransitiveInfoCollectionT>
      Sequence<?> deps, // <AndroidResourcesInfoT>
      boolean neverlink,
      boolean enableDataBinding)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "make_aar",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "resource_info",
            positional = true,
            named = false,
            type = AndroidResourcesInfoApi.class,
            doc =
                "The provider containing processed resources for this target, produced, "
                    + "for example, by merge_resources."),
        @Param(
            name = "asset_info",
            positional = true,
            named = false,
            type = AndroidAssetsInfoApi.class,
            doc =
                "The provider containing processed assets for this target, produced, "
                    + "for example, by merge_assets."),
        @Param(
            name = "library_class_jar",
            positional = true,
            named = false,
            type = FileApi.class,
            doc = "The library class jar."),
        @Param(
            name = "local_proguard_specs",
            type = Sequence.class,
            generic1 = FileApi.class,
            defaultValue = "[]",
            positional = false,
            named = true,
            doc =
                "Files to be used as Proguard specification for this target, which will be"
                    + " inherited in the top-level target."),
        @Param(
            name = "deps",
            type = Sequence.class,
            generic1 = AndroidLibraryAarInfoApi.class,
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
              + " of building the Android Java library.",
      documented = false)
  AndroidLibraryAarInfoT makeAar(
      AndroidDataContextT ctx,
      AndroidResourcesInfoT resourcesInfo,
      AndroidAssetsInfoT assetsInfo,
      FileT libraryClassJar,
      Sequence<?> localProguardSpecs, // <FileT>
      Sequence<?> deps, // <AndroidLibraryAarInfoT>
      boolean neverlink)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "process_aar_import_data",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "resource",
            positional = true,
            named = false,
            type = FileApi.class,
            doc = "The resouce file."),
        @Param(
            name = "assets",
            positional = true,
            named = false,
            type = FileApi.class,
            doc = "The assets file."),
        @Param(
            name = "manifest",
            positional = true,
            named = false,
            type = FileApi.class,
            doc = "The manifest file."),
        @Param(
            name = "deps",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
            named = true,
            positional = false,
            defaultValue = "[]",
            doc = "Targets to inherit asset and resource dependencies from.")
      },
      doc = "Processes assets, resources, and manifest for aar_import targets",
      documented = false)
  Dict<? extends ProviderApi, ? extends StructApi> processAarImportData(
      AndroidDataContextT ctx,
      SpecialFileT resources,
      SpecialFileT assets,
      FileT androidManifest,
      Sequence<?> deps /* <TransitiveInfoCollectionT> */)
      throws InterruptedException, EvalException;

  @SkylarkCallable(
      name = "process_local_test_data",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "manifest",
            positional = false,
            type = FileApi.class,
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
            type = Sequence.class,
            generic1 = FileProviderApi.class,
            named = true,
            doc = "Providers of this target's resources."),
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
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
            type = Dict.class,
            generic1 = String.class,
            named = true,
            doc =
                "A dictionary of values to be overridden in the manifest. You must expand any"
                    + " templates in these values before they are passed to this function."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
            named = true,
            doc =
                "Dependency targets. Providers will be extracted from these dependencies for each"
                    + " type of data."),
        @Param(
            name = "nocompress_extensions",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = String.class,
            named = true,
            doc = "A list of file extensions to leave uncompressed in the resource apk."),
        @Param(
            name = "resource_configuration_filters",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = String.class,
            named = true,
            doc =
                "A list of resource configuration filters, such as 'en' that will limit the"
                    + " resources in the apk to only the ones in the 'en' configuration."),
        @Param(
            name = "densities",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = String.class,
            named = true,
            doc =
                "Densities to filter for when building the apk. A corresponding compatible-screens"
                    + " section will also be added to the manifest if it does not already contain a"
                    + " superset listing."),
      },
      doc =
          "Processes resources, assets, and manifests for android_local_test and returns a dict"
              + " from provider type to the appropriate provider.",
      documented = false)
  Dict<? extends ProviderApi, ? extends StructApi> processLocalTestData(
      AndroidDataContextT ctx,
      Object manifest,
      Sequence<?> resources, // <TransitiveInfoCollectionT>
      Object assets,
      Object assetsDir,
      Object customPackage,
      String aaptVersionString,
      Dict<?, ?> manifestValues, // <String, String>
      Sequence<?> deps, // <TransitiveInfoCollectionT>
      Sequence<?> noCompressExtensions, // <String>
      Sequence<?> resourceConfigurationFilters, // <String>
      Sequence<?> densities) // <String>
      throws InterruptedException, EvalException;

  @SkylarkCallable(
      name = "make_binary_settings",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
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
            type = Sequence.class,
            generic1 = String.class,
            named = true,
            doc =
                "A list of resource configuration filters, such as 'en' that will limit the"
                    + " resources in the apk to only the ones in the 'en' configuration."),
        @Param(
            name = "densities",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
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
            type = Sequence.class,
            generic1 = String.class,
            named = true,
            doc =
                "A list of file extension to leave uncompressed in apk. Templates must be"
                    + " expanded before passing this value in."),
      },
      doc =
          "Returns a wrapper object containing various settings shared across multiple methods for"
              + " processing binary data.",
      documented = false)
  AndroidBinaryDataSettingsApi makeBinarySettings(
      AndroidDataContextT ctx,
      Object shrinkResources,
      Sequence<?> resourceConfigurationFilters, // <String>
      Sequence<?> densities, // <String>
      Sequence<?> noCompressExtensions) // <String>
      throws EvalException;

  @SkylarkCallable(
      name = "process_binary_data",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = FileProviderApi.class,
            named = true,
            doc = "Providers of this target's resources."),
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
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
            type = FileApi.class,
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
            type = Dict.class,
            generic1 = String.class,
            named = true,
            doc =
                "A dictionary of values to be overridden in the manifest. You must expand any"
                    + " templates in the values before calling this function."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
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
            type = AndroidBinaryDataSettingsApi.class,
            noneable = true,
            defaultValue = "None",
            positional = false,
            named = true,
            doc =
                "Settings common to various binary processing methods, created with"
                    + " make_binary_data_settings."),
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
              + " appropriate providers.",
      documented = false)
  AndroidBinaryDataInfoT processBinaryData(
      AndroidDataContextT ctx,
      Sequence<?> resources, // <TransitiveInfoCollectionT>
      Object assets,
      Object assetsDir,
      Object manifest,
      Object customPackage,
      Dict<?, ?> manifestValues, // <String, String>
      Sequence<?> deps, // <TransitiveInfoCollectionT>
      String manifestMerger,
      Object maybeSettings,
      boolean crunchPng,
      boolean dataBindingEnabled)
      throws InterruptedException, EvalException;

  @SkylarkCallable(
      name = "shrink_data_apk",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = AndroidDataContextApi.class,
            doc = "The Android data context object for this target."),
        @Param(
            name = "binary_data_info",
            positional = true,
            named = false,
            type = AndroidBinaryDataInfoApi.class,
            doc = "The Info about the binary to shrink, as produced by process_binary_data."),
        @Param(
            name = "proguard_output_jar",
            positional = true,
            named = false,
            type = FileApi.class,
            doc = "The proguard jar output file."),
        @Param(
            name = "proguard_mapping",
            positional = true,
            named = false,
            type = FileApi.class,
            doc = "The proguard mapping output file."),
        @Param(
            name = "binary_settings",
            type = AndroidBinaryDataSettingsApi.class,
            noneable = true,
            defaultValue = "None",
            positional = false,
            named = true,
            doc =
                "Settings common to various binary processing methods, created with"
                    + " make_binary_data_settings."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
            named = true,
            doc =
                "Dependency targets. Providers will be extracted from these dependencies for each"
                    + " type of data."),
        @Param(
            name = "proguard_specs",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
            defaultValue = "[]",
            positional = false,
            named = true,
            doc =
                "Files to be used as Proguard specification for this target, which will be"
                    + " inherited in the top-level target."),
        @Param(
            name = "extra_proguard_specs,",
            type = Sequence.class,
            generic1 = TransitiveInfoCollectionApi.class,
            defaultValue = "[]",
            positional = false,
            named = true,
            doc =
                "Additional proguard specs that should be added for top-level targets. This  value"
                    + " is controlled by Java configuration.")
      },
      doc =
          "Possibly shrinks the data APK by removing resources that were marked as unused during"
              + " proguarding.",
      documented = false)
  AndroidBinaryDataInfoT shrinkDataApk(
      AndroidDataContextT ctx,
      AndroidBinaryDataInfoT binaryDataInfo,
      FileT proguardOutputJar,
      FileT proguardMapping,
      Object maybeSettings,
      Sequence<?> deps, // <TransitiveInfoCollectionT>
      Sequence<?> localProguardSpecs, // <TransitiveInfoCollectionT>
      Sequence<?> extraProguardSpecs) // <TransitiveInfoCollectionT>
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "resources_from_validated_res",
      allowReturnNones = true,
      doc = "Returns an Artifact containing a zip of merged resources.",
      documented = false,
      parameters = {
        @Param(
            name = "validated_res",
            doc = "The validated Android resources.",
            positional = true,
            named = false,
            type = ValidatedAndroidDataApi.class)
      })
  FileT resourcesFromValidatedRes(ValidatedAndroidDataT resources);
}
