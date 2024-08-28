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
package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FileProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.TransitiveInfoCollectionApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Starlark-visible methods for working with Android data (manifests, resources, and assets). */
@StarlarkBuiltin(
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

  @StarlarkMethod(
      name = "assets_from_deps",
      parameters = {
        @Param(
            name = "deps",
            defaultValue = "[]",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = AndroidAssetsInfoApi.class)
            },
            positional = false,
            named = true,
            doc = "Dependencies to inherit assets from."),
        @Param(
            name = "neverlink",
            defaultValue = "False",
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

  @StarlarkMethod(
      name = "resources_from_deps",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            doc = "The Android data context object for this target."),
        @Param(
            name = "deps",
            defaultValue = "[]",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = AndroidResourcesInfoApi.class)
            },
            positional = false,
            named = true,
            doc = "Dependencies to inherit resources from."),
        @Param(
            name = "assets",
            defaultValue = "[]",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = AndroidAssetsInfoApi.class),
            },
            positional = false,
            named = true,
            doc = "Dependencies to inherit assets from."),
        @Param(
            name = "neverlink",
            defaultValue = "False",
            positional = false,
            named = true,
            doc =
                "Defaults to False. If true, resources will not be exposed to targets that depend"
                    + " on them."),
        @Param(
            name = "custom_package",
            positional = false,
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

  @StarlarkMethod(
      name = "stamp_manifest",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            doc = "The Android data context object for this target."),
        @Param(
            name = "manifest",
            positional = false,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = FileApi.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            doc = "The manifest to stamp. If not passed, a dummy manifest will be generated."),
        @Param(
            name = "custom_package",
            positional = false,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = NoneType.class),
            },
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

  @StarlarkMethod(
      name = "merge_assets",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            doc = "The Android data context object for this target."),
        @Param(
            name = "assets",
            positional = false,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = TransitiveInfoCollectionApi.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            doc =
                "Targets containing raw assets for this target. If passed, 'assets_dir' must also"
                    + " be passed."),
        @Param(
            name = "assets_dir",
            positional = false,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = NoneType.class),
            },
            named = true,
            doc =
                "Directory the assets are contained in. Must be passed if and only if 'assets' is"
                    + " passed. This path will be split off of the asset paths on the device."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = AndroidAssetsInfoApi.class)
            },
            named = true,
            doc =
                "Providers containing assets from dependencies. These assets will be merged"
                    + " together with each other and this target's assets."),
        @Param(
            name = "neverlink",
            positional = false,
            defaultValue = "False",
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

  @StarlarkMethod(
      name = "merge_res",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            doc = "The Android data context object for this target."),
        @Param(
            name = "manifest",
            positional = true,
            named = false,
            doc =
                "The provider of this target's manifest. This provider is produced by, "
                    + "for example, stamp_android_manifest."),
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileProviderApi.class)},
            named = true,
            doc = "Providers of this target's resources."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = AndroidResourcesInfoApi.class),
            },
            named = true,
            doc =
                "Targets containing raw resources from dependencies. These resources will be merged"
                    + " together with each other and this target's resources."),
        @Param(
            name = "validation_resource_apks",
            positional = false,
            defaultValue = "[]",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = FileProviderApi.class),
            },
            named = true,
            doc =
                "List of resource only APK files to be used for validation only. Not fully"
                    + " supported in the native resource pipeline."),
        @Param(
            name = "neverlink",
            positional = false,
            defaultValue = "False",
            named = true,
            doc =
                "Defaults to False. If passed as True, these resources will not be inherited by"
                    + " targets that depend on this one."),
        @Param(
            name = "enable_data_binding",
            positional = false,
            defaultValue = "False",
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
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  ValidatedAndroidDataT mergeRes(
      AndroidDataContextT ctx,
      AndroidManifestInfoT manifest,
      Sequence<?> resources, // <TransitiveInfoCollectionT>
      Sequence<?> deps, // <AndroidResourcesInfoT>
      Sequence<?> resApkDeps, // <FileT>
      boolean neverlink,
      boolean enableDataBinding)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "merge_resources",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            doc = "The Android data context object for this target."),
        @Param(
            name = "manifest",
            positional = true,
            named = false,
            doc =
                "The provider of this target's manifest. This provider is produced by, "
                    + "for example, stamp_android_manifest."),
        @Param(
            name = "resources",
            positional = false,
            defaultValue = "[]",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileProviderApi.class)},
            named = true,
            doc = "Providers of this target's resources."),
        @Param(
            name = "deps",
            positional = false,
            defaultValue = "[]",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = AndroidResourcesInfoApi.class)
            },
            named = true,
            doc =
                "Targets containing raw resources from dependencies. These resources will be merged"
                    + " together with each other and this target's resources."),
        @Param(
            name = "neverlink",
            positional = false,
            defaultValue = "False",
            named = true,
            doc =
                "Defaults to False. If passed as True, these resources will not be inherited by"
                    + " targets that depend on this one."),
        @Param(
            name = "enable_data_binding",
            positional = false,
            defaultValue = "False",
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
      throws EvalException, InterruptedException, RuleErrorException;

  @StarlarkMethod(
      name = "make_aar",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            doc = "The Android data context object for this target."),
        @Param(
            name = "resource_info",
            positional = true,
            named = false,
            doc =
                "The provider containing processed resources for this target, produced, "
                    + "for example, by merge_resources."),
        @Param(
            name = "asset_info",
            positional = true,
            named = false,
            doc =
                "The provider containing processed assets for this target, produced, "
                    + "for example, by merge_assets."),
        @Param(
            name = "library_class_jar",
            positional = true,
            named = false,
            doc = "The library class jar."),
        @Param(
            name = "local_proguard_specs",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
            defaultValue = "[]",
            positional = false,
            named = true,
            doc =
                "Files to be used as Proguard specification for this target, which will be"
                    + " inherited in the top-level target."),
        @Param(
            name = "deps",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = AndroidLibraryAarInfoApi.class)
            },
            defaultValue = "[]",
            positional = false,
            named = true,
            doc = "Dependant AAR providers used to build this AAR."),
        @Param(
            name = "neverlink",
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

  @StarlarkMethod(
      name = "process_aar_import_data",
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            doc = "The Android data context object for this target."),
        @Param(name = "resource", positional = true, named = false, doc = "The resource file."),
        @Param(name = "assets", positional = true, named = false, doc = "The assets file."),
        @Param(name = "manifest", positional = true, named = false, doc = "The manifest file."),
        @Param(
            name = "deps",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = TransitiveInfoCollectionApi.class)
            },
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
      throws InterruptedException, EvalException, RuleErrorException;
}
