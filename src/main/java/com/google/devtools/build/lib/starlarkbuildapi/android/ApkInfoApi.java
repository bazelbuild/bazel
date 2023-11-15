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
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
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

/** A provider for targets that produce an apk file. */
@StarlarkBuiltin(
    name = "ApkInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "APKs provided by a rule",
    documented = false,
    category = DocCategory.PROVIDER)
public interface ApkInfoApi<FileT extends FileApi> extends StructApi {

  /** Name of this info object. */
  String NAME = "ApkInfo";

  /** Returns the APK file built in the transitive closure. */
  @StarlarkMethod(
      name = "signed_apk",
      doc = "Returns a signed APK built from the target.",
      documented = false,
      structField = true)
  FileT getApk();

  /** Returns the unsigned APK file built in the transitive closure. */
  @StarlarkMethod(
      name = "unsigned_apk",
      doc = "Returns a unsigned APK built from the target.",
      documented = false,
      structField = true)
  FileT getUnsignedApk();

  /** Returns the deploy jar used to build the APK. */
  @StarlarkMethod(
      name = "deploy_jar",
      doc = "Returns the deploy jar used to build the APK.",
      documented = false,
      structField = true)
  FileT getDeployJar();

  /** Returns the coverage metadata artifact generated in the transitive closure. */
  @StarlarkMethod(
      name = "coverage_metadata",
      doc = "Returns the coverage metadata artifact generated in the transitive closure.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  FileT getCoverageMetadata();

  /**
   * Returns keystore that was used to sign the APK.
   *
   * <p>Prefer using getSigningKeys(), this method is deprecated.
   */
  @StarlarkMethod(
      name = "keystore",
      doc = "Returns a keystore that was used to sign the APK. Deprecated: prefer signing_keys.",
      documented = false,
      structField = true)
  FileT getKeystore();

  /** Returns a list of signing keystores that were used to sign the APK */
  @StarlarkMethod(
      name = "signing_keys",
      doc = "Returns a list of signing keystores that were used to sign the APK.",
      documented = false,
      structField = true)
  ImmutableList<FileT> getSigningKeys();

  /** Returns the signing lineage file, if present, that was used to sign the APK */
  @StarlarkMethod(
      name = "signing_lineage",
      doc = "Returns the signing lineage file, if present, that was used to sign the APK.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  FileT getSigningLineage();

  /** Returns the minimum API version for signing the APK with key rotation. */
  @StarlarkMethod(
      name = "signing_min_v3_rotation_api_version",
      doc = "Returns the minimum API version for signing the APK with key rotation.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getSigningMinV3RotationApiVersion();

  /** Provider for {@link ApkInfoApi}. */
  @StarlarkBuiltin(
      name = "ApkInfoApiProvider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface ApkInfoApiProvider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = "ApkInfo",
        doc = "The <code>ApkInfo<code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "signed_apk",
              doc = "The signed APK file.",
              allowedTypes = {
                @ParamType(type = FileApi.class),
              },
              named = true),
          @Param(
              name = "unsigned_apk",
              doc = "The unsigned APK file.",
              allowedTypes = {
                @ParamType(type = FileApi.class),
              },
              named = true),
          @Param(
              name = "deploy_jar",
              doc = "The deploy jar file.",
              allowedTypes = {
                @ParamType(type = FileApi.class),
              },
              named = true),
          @Param(
              name = "coverage_metadata",
              doc = "The coverage metadata file generated in the transitive closure.",
              allowedTypes = {@ParamType(type = FileApi.class), @ParamType(type = NoneType.class)},
              named = true),
          @Param(
              name = "merged_manifest",
              doc = "The merged manifest file.",
              allowedTypes = {
                @ParamType(type = FileApi.class),
              },
              named = true),
          @Param(
              name = "signing_keys",
              doc = "The list of signing keys used to sign the APK.",
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = FileApi.class)},
              named = true),
          @Param(
              name = "signing_lineage",
              doc = "The signing lineage file. If present, that was used to sign the APK",
              allowedTypes = {@ParamType(type = FileApi.class), @ParamType(type = NoneType.class)},
              named = true),
          @Param(
              name = "signing_min_v3_rotation_api_version",
              doc = "The minimum API version for signing the APK with key rotation.",
              allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)},
              named = true),
        },
        selfCall = true)
    @StarlarkConstructor
    ApkInfoApi<FileT> createInfo(
        FileT signedApk,
        FileT unsignedApk,
        FileT deployJar,
        Object coverageMetadata,
        FileT mergedManifest,
        Sequence<?> signingKeys, // <Artifact> expected
        Object signingLineage,
        Object signingMinV3RotationApiVersion)
        throws EvalException;
  }
}
