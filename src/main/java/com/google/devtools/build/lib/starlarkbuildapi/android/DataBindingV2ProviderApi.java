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
package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/**
 * An interface for a provider that exposes the use of <a
 * href="https://developer.android.com/topic/libraries/data-binding/index.html">data binding</a>.
 */
@StarlarkBuiltin(
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
  @StarlarkBuiltin(
      name = "LabelJavaPackagePair",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  final class LabelJavaPackagePair implements StarlarkValue {

    public static final Depset.ElementType TYPE = Depset.ElementType.of(LabelJavaPackagePair.class);

    private final String label;
    private final String javaPackage;

    public LabelJavaPackagePair(String label, String javaPackage) {
      this.label = label;
      this.javaPackage = javaPackage;
    }

    @StarlarkMethod(name = "label", structField = true, doc = "", documented = false)
    public String getLabel() {
      return label;
    }

    @StarlarkMethod(name = "java_package", structField = true, doc = "", documented = false)
    public String getJavaPackage() {
      return javaPackage;
    }
  }

  /** Name of this info object. */
  String NAME = "DataBindingV2Info";

  /**
   * Returns the setter store files from this rule. This is a Depset to support multiple
   * android_libraries in the exports attribute, where the providers from exports are merged into a
   * single provider. In a rule without exports, this will be at most 1 file.
   */
  @StarlarkMethod(name = "setter_stores", structField = true, doc = "", documented = false)
  Depset /*<T>*/ getSetterStoresForStarlark();

  /**
   * Returns the class info files from this rule. This is a Depset to support multiple
   * android_libraries in the exports attribute, where the providers from exports are merged into a
   * single provider. In a rule without exports, this will be at most 1 file.
   */
  @StarlarkMethod(name = "class_infos", structField = true, doc = "", documented = false)
  Depset /*<T>*/ getClassInfosForStarlark();

  /** Returns the BR files from this rule and its dependencies. */
  @StarlarkMethod(name = "transitive_br_files", structField = true, doc = "", documented = false)
  Depset /*<T>*/ getTransitiveBRFilesForStarlark();

  /**
   * Returns a NestedSet containing the label and java package for this rule and its transitive
   * dependencies.
   */
  @StarlarkMethod(
      name = "transitive_label_and_java_packages",
      structField = true,
      doc = "",
      documented = false)
  Depset /*<LabelJavaPackagePair>*/ getTransitiveLabelAndJavaPackagesForStarlark();

  /** Returns the label and java package for this rule and any rules that this rule exports. */
  @StarlarkMethod(
      name = "label_and_java_packages",
      structField = true,
      allowReturnNones = true,
      doc = "",
      documented = false)
  @Nullable
  ImmutableList<LabelJavaPackagePair> getLabelAndJavaPackages();

  /** The provider implementing this can construct the DataBindingV2Info provider. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>DataBindingV2Info</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "setter_store_file",
              doc = "The setter_stores.bin files .",
              positional = false,
              named = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "class_info_file",
              doc = "The class_info files for this rule.",
              positional = false,
              named = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "br_file",
              doc = "The br file for this rule.",
              positional = false,
              named = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = FileApi.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "label",
              doc = "The label of the current rule.",
              positional = false,
              named = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = String.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "java_package",
              doc = "The java package of the current rule.",
              positional = false,
              named = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = String.class),
                @ParamType(type = NoneType.class),
              }),
          @Param(
              name = "databinding_v2_providers_in_deps",
              doc = "The DatabindingV2Provider instances from dependencies.",
              positional = false,
              named = true,
              defaultValue = "[]",
              allowedTypes = {
                @ParamType(type = Sequence.class, generic1 = DataBindingV2ProviderApi.class)
              }),
          @Param(
              name = "databinding_v2_providers_in_exports",
              doc = "The DatabindingV2Provider instances from exports.",
              positional = false,
              named = true,
              defaultValue = "[]",
              allowedTypes = {
                @ParamType(type = Sequence.class, generic1 = DataBindingV2ProviderApi.class)
              }),
        },
        selfCall = true)
    @StarlarkConstructor
    DataBindingV2ProviderApi<FileT> createInfo(
        Object setterStoreFile,
        Object classInfoFile,
        Object brFile,
        Object label,
        Object javaPackage,
        Sequence<?> databindingV2ProvidersInDeps, // <DataBindingV2ProviderApi<FileT>>
        Sequence<?> databindingV2ProvidersInExports /* <DataBindingV2ProviderApi<FileT>> */)
        throws EvalException;
  }
}
