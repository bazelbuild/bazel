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

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** A target that can provide the aar artifact of Android libraries */
@SkylarkModule(
    name = "AndroidLibraryAarInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Android AARs provided by a library rule and its dependencies",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER)
public interface AndroidLibraryAarInfoApi<FileT extends FileApi> extends StructApi {

  /** The name of the provider for this info object. */
  String NAME = "AndroidLibraryAarInfo";

  @SkylarkCallable(
      name = "aar",
      doc = "",
      documented = false,
      structField = true,
      allowReturnNones = true)
  FileT getAarArtifact();

  @SkylarkCallable(
      name = "transitive_aar_artifacts",
      doc = "",
      documented = false,
      structField = true)
  NestedSet<FileT> getTransitiveAarArtifacts();
}
