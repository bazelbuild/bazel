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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import javax.annotation.Nullable;

/** Provides information about transitive Android assets. */
@SkylarkModule(
    name = "AndroidAssetsInfo",
    doc = "Information about the Android assets provided by a rule.",
    category = SkylarkModuleCategory.PROVIDER)
public interface AndroidAssetsInfoApi extends StructApi {

  @SkylarkCallable(
      name = "validation_result",
      structField = true,
      allowReturnNones = true,
      doc =
          "If not None, represents the output of asset merging and validation for this target. The"
              + " action to merge and validate assets is not run be default; to force it, add this"
              + " artifact to your target's outputs. The validation action is somewhat expensive -"
              + " in native code, this artifact is added to the top-level output group (so"
              + " validation is only done if the target is requested on the command line). The"
              + " contents of this artifact are subject to change and should not be relied upon.")
  @Nullable
  FileApi getValidationResult();

  /** Returns the local assets for the target. */
  @SkylarkCallable(
      name = "local_assets",
      doc = "Returns the local assets for the target.",
      allowReturnNones = true,
      structField = true)
  ImmutableList<? extends FileApi> getLocalAssets();

  /** Returns the local asset dir for the target. */
  @SkylarkCallable(
      name = "local_asset_dir",
      doc = "Returns the local asset directory for the target.",
      allowReturnNones = true,
      structField = true)
  String getLocalAssetDir();
}
