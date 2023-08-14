// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Configuration fragment for Android rules. */
@StarlarkBuiltin(
    name = "android",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed. "
            + "A configuration fragment for Android.",
    documented = false,
    category = DocCategory.CONFIGURATION_FRAGMENT)
public interface AndroidConfigurationApi extends StarlarkValue {

  @StarlarkMethod(
      name = "android_cpu",
      structField = true,
      doc = "The Android target CPU.",
      documented = false)
  String getCpu();

  @StarlarkMethod(name = "use_incremental_dexing", structField = true, doc = "", documented = false)
  boolean useIncrementalDexing();

  @StarlarkMethod(
      name = "incremental_dexing_shards_after_proguard",
      structField = true,
      doc = "",
      documented = false)
  int incrementalDexingShardsAfterProguard();

  @StarlarkMethod(
      name = "incremental_dexing_use_dex_sharder",
      structField = true,
      doc = "",
      documented = false)
  boolean incrementalDexingUseDexSharder();

  @StarlarkMethod(
      name = "incremental_dexing_after_proguard_by_default",
      structField = true,
      doc = "",
      documented = false)
  boolean incrementalDexingAfterProguardByDefault();

  @StarlarkMethod(name = "apk_signing_method_v1", structField = true, doc = "", documented = false)
  boolean apkSigningMethodV1();

  @StarlarkMethod(name = "apk_signing_method_v2", structField = true, doc = "", documented = false)
  boolean apkSigningMethodV2();

  @StarlarkMethod(
      name = "apk_signing_method_v4",
      structField = true,
      doc = "",
      documented = false,
      allowReturnNones = true)
  @Nullable
  Boolean apkSigningMethodV4();

  @StarlarkMethod(name = "assume_min_sdk_version", structField = true, doc = "", documented = false)
  boolean assumeMinSdkVersion();

  @StarlarkMethod(
      name = "get_dexopts_supported_in_incremental_dexing",
      structField = true,
      doc = "",
      documented = false)
  ImmutableList<String> getDexoptsSupportedInIncrementalDexing();

  @StarlarkMethod(
      name = "get_dexopts_supported_in_dex_merger",
      structField = true,
      doc = "",
      documented = false)
  ImmutableList<String> getDexoptsSupportedInDexMerger();

  @StarlarkMethod(
      name = "get_dexopts_supported_in_dex_sharder",
      structField = true,
      doc = "",
      documented = false)
  ImmutableList<String> getDexoptsSupportedInDexSharder();

  @StarlarkMethod(
      name = "get_target_dexopts_that_prevent_incremental_dexing",
      structField = true,
      doc = "",
      documented = false)
  ImmutableList<String> getTargetDexoptsThatPreventIncrementalDexing();

  @StarlarkMethod(name = "desugar_java8", structField = true, doc = "", documented = false)
  boolean desugarJava8();

  @StarlarkMethod(name = "desugar_java8_libs", structField = true, doc = "", documented = false)
  boolean desugarJava8Libs();

  @StarlarkMethod(name = "check_desugar_deps", structField = true, doc = "", documented = false)
  boolean checkDesugarDeps();

  @StarlarkMethod(
      name = "use_rex_to_compress_dex_files",
      structField = true,
      doc = "",
      documented = false)
  boolean useRexToCompressDexFiles();

  @StarlarkMethod(
      name = "use_android_resource_shrinking",
      structField = true,
      doc = "",
      documented = false)
  boolean useAndroidResourceShrinking();

  @StarlarkMethod(
      name = "use_android_resource_cycle_shrinking",
      structField = true,
      doc = "",
      documented = false)
  boolean useAndroidResourceCycleShrinking();

  @StarlarkMethod(
      name = "use_android_resource_path_shortening",
      structField = true,
      doc = "",
      documented = false)
  boolean useAndroidResourcePathShortening();

  @StarlarkMethod(
      name = "use_android_resource_name_obfuscation",
      structField = true,
      doc = "",
      documented = false)
  boolean useAndroidResourceNameObfuscation();

  @StarlarkMethod(
      name = "use_single_jar_apk_builder",
      structField = true,
      doc = "",
      documented = false)
  boolean useSingleJarApkBuilder();

  @StarlarkMethod(name = "use_parallel_dex2oat", structField = true, doc = "", documented = false)
  boolean useParallelDex2Oat();

  @StarlarkMethod(
      name = "break_build_on_parallel_dex2oat_failure",
      structField = true,
      doc = "",
      documented = false)
  boolean breakBuildOnParallelDex2OatFailure();

  @StarlarkMethod(
      name = "compress_java_resources",
      structField = true,
      doc = "",
      documented = false)
  boolean compressJavaResources();

  @StarlarkMethod(
      name = "get_exports_manifest_default",
      structField = true,
      doc = "",
      documented = false)
  boolean getExportsManifestDefault();

  @StarlarkMethod(name = "manifest_merger", structField = true, doc = "", documented = false)
  String getManifestMergerValue();

  @StarlarkMethod(
      name = "omit_resources_info_provider_from_android_binary",
      structField = true,
      doc = "",
      documented = false)
  boolean omitResourcesInfoProviderFromAndroidBinary();

  @StarlarkMethod(
      name = "fixed_resource_neverlinking",
      structField = true,
      doc = "",
      documented = false)
  boolean fixedResourceNeverlinking();

  @StarlarkMethod(
      name = "check_for_migration_tag",
      structField = true,
      doc = "",
      documented = false)
  boolean checkForMigrationTag();

  @StarlarkMethod(
      name = "get_one_version_enforcement_use_transitive_jars_for_binary_under_test",
      structField = true,
      doc = "",
      documented = false)
  boolean getOneVersionEnforcementUseTransitiveJarsForBinaryUnderTest();

  @StarlarkMethod(name = "use_databinding_v2", structField = true, doc = "", documented = false)
  boolean useDataBindingV2();

  @StarlarkMethod(
      name = "android_databinding_use_v3_4_args",
      structField = true,
      doc = "",
      documented = false)
  boolean useDataBindingUpdatedArgs();

  @StarlarkMethod(
      name = "android_databinding_use_androidx",
      structField = true,
      doc = "",
      documented = false)
  boolean useDataBindingAndroidX();

  @StarlarkMethod(
      name = "persistent_busybox_tools",
      structField = true,
      doc = "",
      documented = false)
  boolean persistentBusyboxTools();

  @StarlarkMethod(
      name = "persistent_multiplex_busybox_tools",
      structField = true,
      doc = "",
      documented = false)
  boolean persistentMultiplexBusyboxTools();

  @StarlarkMethod(
      name = "persistent_android_dex_desugar",
      structField = true,
      doc = "",
      documented = false)
  boolean persistentDexDesugar();

  @StarlarkMethod(
      name = "persistent_multiplex_android_dex_desugar",
      structField = true,
      doc = "",
      documented = false)
  boolean persistentMultiplexDexDesugar();

  @StarlarkMethod(
      name = "get_output_directory_name",
      structField = true,
      doc = "",
      documented = false)
  String getOutputDirectoryName();

  @StarlarkMethod(
      name = "incompatible_use_toolchain_resolution",
      structField = true,
      doc = "",
      documented = false)
  boolean incompatibleUseToolchainResolution();

  @StarlarkMethod(name = "hwasan", structField = true, doc = "", documented = false)
  boolean isHwasan();

  @StarlarkMethod(
      name = "filter_library_jar_with_program_jar",
      structField = true,
      doc = "",
      documented = false)
  boolean filterLibraryJarWithProgramJar();
}
