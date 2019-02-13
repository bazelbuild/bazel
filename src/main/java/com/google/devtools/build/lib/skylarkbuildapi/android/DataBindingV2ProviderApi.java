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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import javax.annotation.Nullable;

/**
 * An interface for a provider that exposes the use of <a
 * href="https://developer.android.com/topic/libraries/data-binding/index.html">data binding</a>.
 */
@SkylarkModule(
    name = "DataBindingV2Info",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface DataBindingV2ProviderApi<T extends FileApi> extends StructApi {

  /**
   * A pair of label as a string and the Java package for that label as determined by the Android
   * rules. This is for reporting a useful error message if multiple android_library rules with the
   * same Java package end up in the same android_binary.
   */
  @AutoValue
  public abstract class LabelJavaPackagePair {

    public static LabelJavaPackagePair create(String label, String javaPackage) {
      return new AutoValue_DataBindingV2ProviderApi_LabelJavaPackagePair(label, javaPackage);
    }

    public abstract String label();

    public abstract String javaPackage();
  }

  /** Name of this info object. */
  public static final String NAME = "DataBindingV2Info";

  /**
   * Returns the setter store files from this rule. This is a list to support multiple
   * android_libraries in the exports attribute, where the providers from exports are merged into a
   * single provider. In a rule without exports, this will be at most 1 file.
   */
  @SkylarkCallable(name = "setter_stores", structField = true, doc = "", documented = false)
  ImmutableList<T> getSetterStores();

  /**
   * Returns the class info files from this rule. This is a list to support multiple
   * android_libraries in the exports attribute, where the providers from exports are merged into a
   * single provider. In a rule without exports, this will be at most 1 file.
   */
  @SkylarkCallable(name = "class_infos", structField = true, doc = "", documented = false)
  ImmutableList<T> getClassInfos();

  /** Returns the BR files from this rule and its dependencies. */
  @SkylarkCallable(name = "transitive_br_files", structField = true, doc = "", documented = false)
  NestedSet<T> getTransitiveBRFiles();

  /**
   * Returns a NestedSet containing the label and java package for this rule and its transitive
   * dependencies.
   * */
  @SkylarkCallable(
      name = "transitive_label_and_java_packages",
      structField = true,
      doc = "",
      documented = false)
  NestedSet<LabelJavaPackagePair> getTransitiveLabelAndJavaPackages();

  /**
   * Returns the label and java package for this rule and any rules that this rule exports.
   */
  @SkylarkCallable(
      name = "label_and_java_packages",
      structField = true,
      doc = "",
      documented = false)
  @Nullable
  ImmutableList<LabelJavaPackagePair> getLabelAndJavaPackages();

  /** The provider implementing this can construct the DataBindingV2Info provider. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  public interface Provider<F extends FileApi> extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>DataBindingV2Info</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "setter_store_file",
              doc = "The setter_stores.bin files .",
              positional = true,
              named = false,
              type = FileApi.class),
          @Param(
              name = "class_info_file",
              doc = "The class_info files for this rule.",
              positional = true,
              named = false,
              type = FileApi.class),
          @Param(
              name = "br_file",
              doc = "The br file for this rule.",
              positional = true,
              named = false,
              type = FileApi.class),
          @Param(
              name = "label",
              doc = "The label of the current rule.",
              positional = true,
              named = false,
              type = String.class),
          @Param(
              name = "java_package",
              doc = "The java package of the current rule.",
              positional = true,
              named = false,
              type = String.class),
          @Param(
              name = "databinding_v2_providers_in_deps",
              doc = "The DatabindingV2Provider instances from dependencies.",
              positional = true,
              named = false,
              type = SkylarkList.class,
              generic1 = DataBindingV2ProviderApi.class),
          @Param(
              name = "databinding_v2_providers_in_exports",
              doc = "The DatabindingV2Provider instances from exports.",
              positional = true,
              named = false,
              type = SkylarkList.class,
              generic1 = DataBindingV2ProviderApi.class),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = DataBindingV2ProviderApi.class)
    DataBindingV2ProviderApi<F> createInfo(
        F setterStoreFile,
        F classInfoFile,
        F brFile,
        String label,
        String javaPackage,
        SkylarkList<DataBindingV2ProviderApi<F>> databindingV2ProvidersInDeps,
        SkylarkList<DataBindingV2ProviderApi<F>> databindingV2ProvidersInExports)
        throws EvalException;
  }
}