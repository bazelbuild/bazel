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
package com.google.devtools.build.lib.skylarkbuildapi.android;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;

/** A provider that supplies resource information from its transitive closure. */
@SkylarkModule(
    name = "AndroidResourcesInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Android resources provided by a rule",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER)
public interface AndroidResourcesInfoApi<
        FileT extends FileApi,
        ValidatedAndroidDataT extends ValidatedAndroidDataApi,
        AndroidManifestInfoT extends AndroidManifestInfoApi<FileT>>
    extends StructApi {

  /**
   * Name of this info object.
   */
  public static String NAME = "AndroidResourcesInfo";

  /** Returns the label that is associated with this piece of information. */
  @SkylarkCallable(
      name = "label",
      doc = "Returns the label that is associated with this piece of information.",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  Label getLabel();

  @SkylarkCallable(
      name = "manifest",
      doc = "",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  AndroidManifestInfoT getManifest();

  /** Returns the compiletime r.txt file for the target. */
  @SkylarkCallable(
      name = "compiletime_r_txt",
      doc =
          "A txt file containing compiled resource file information for this target. This is a"
              + " stubbed out compiletime file and should not be built into APKs, inherited from"
              + " dependencies, or used at runtime.",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  FileT getRTxt();

  /** Returns the transitive ResourceContainers for the label. */
  @SkylarkCallable(
      name = "transitive_android_resources",
      doc = "Returns the transitive ResourceContainers for the label.",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<ValidatedAndroidDataT> getTransitiveAndroidResources();

  /** Returns the immediate ResourceContainers for the label. */
  @SkylarkCallable(
      name = "direct_android_resources",
      doc = "Returns the immediate ResourceContainers for the label.",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<ValidatedAndroidDataT> getDirectAndroidResources();

  @SkylarkCallable(
      name = "transitive_resources",
      doc = "",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<FileT> getTransitiveResources();

  @SkylarkCallable(
      name = "transitive_manifests",
      doc = "",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<FileT> getTransitiveManifests();

  @SkylarkCallable(
      name = "transitive_aapt2_r_txt",
      doc = "",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<FileT> getTransitiveAapt2RTxt();

  @SkylarkCallable(
      name = "transitive_symbols_bin",
      doc = "",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<FileT> getTransitiveSymbolsBin();

  @SkylarkCallable(
      name = "transitive_compiled_symbols",
      doc = "",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<FileT> getTransitiveCompiledSymbols();

  @SkylarkCallable(
      name = "transitive_static_lib",
      doc = "",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<FileT> getTransitiveStaticLib();

  @SkylarkCallable(
      name = "transitive_r_txt",
      doc = "",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  NestedSet<FileT> getTransitiveRTxt();

  /** Provider for {@link AndroidResourcesInfoApi}. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  public interface AndroidResourcesInfoApiProvider<
          FileT extends FileApi,
          ValidatedAndroidDataT extends ValidatedAndroidDataApi,
          AndroidManifestInfoT extends AndroidManifestInfoApi<FileT>>
      extends ProviderApi {

    @SkylarkCallable(
        name = "AndroidResourcesInfo",
        doc = "The <code>AndroidResourcesInfo</code> constructor.",
        documented = false,
        enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS,
        parameters = {
          @Param(
              name = "label",
              doc = "A label of the target.",
              positional = true,
              named = false,
              type = Label.class),
          @Param(
              name = "manifest",
              positional = true,
              named = false,
              type = AndroidManifestInfoApi.class),
          @Param(name = "r_txt", positional = true, named = false, type = FileApi.class),
          @Param(
              name = "transitive_android_resources",
              doc =
                  "A depset of ValidatedAndroidData of Android Resources in the transitive "
                      + "closure.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = ValidatedAndroidDataApi.class),
          @Param(
              name = "direct_android_resources",
              doc = "A depset of ValidatedAndroidData of Android Resources for the target.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = ValidatedAndroidDataApi.class),
          @Param(
              name = "transitive_resources",
              doc = "A depset of Artifacts of Android Resource files in the transitive closure.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = FileApi.class),
          @Param(
              name = "transitive_manifests",
              doc = "A depset of Artifacts of Android Manifests in the transitive closure.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = FileApi.class),
          @Param(
              name = "transitive_aapt2_r_txt",
              doc = "A depset of Artifacts of Android AAPT2 R.txt files in the transitive closure.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = FileApi.class),
          @Param(
              name = "transitive_symbols_bin",
              doc = "A depset of Artifacts of Android symbols files in the transitive closure.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = FileApi.class),
          @Param(
              name = "transitive_compiled_symbols",
              doc =
                  "A depset of Artifacts of Android compiled symbols files in the transitive "
                      + "closure.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = FileApi.class),
          @Param(
              name = "transitive_static_lib",
              doc = "A depset of Artifacts of static lib files in the transitive closure.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = FileApi.class),
          @Param(
              name = "transitive_r_txt",
              doc = "A depset of Artifacts of Android AAPT R.txt files in the transitive closure.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = FileApi.class),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = AndroidResourcesInfoApi.class, receiverNameForDoc = NAME)
    AndroidResourcesInfoApi<FileT, ValidatedAndroidDataT, AndroidManifestInfoT> createInfo(
        Label label,
        AndroidManifestInfoT manifest,
        FileT rTxt,
        SkylarkNestedSet transitiveAndroidResources,
        SkylarkNestedSet directAndroidResources,
        SkylarkNestedSet transitiveResources,
        SkylarkNestedSet transitiveManifests,
        SkylarkNestedSet transitiveAapt2RTxt,
        SkylarkNestedSet transitiveSymbolsBin,
        SkylarkNestedSet transitiveCompiledSymbols,
        SkylarkNestedSet transitiveStaticLib,
        SkylarkNestedSet transitiveRTxt)
        throws EvalException;
  }
}
