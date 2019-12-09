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

package com.google.devtools.build.lib.skylarkbuildapi.apple;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import javax.annotation.Nullable;

/**
 * An interface representing an info type containing information about an Apple dynamic framework.
 */
@SkylarkModule(
    name = "AppleDynamicFramework",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "A provider containing information about an Apple dynamic framework.")
public interface AppleDynamicFrameworkInfoApi<FileApiT extends FileApi> extends StructApi {

  /**
   * Returns the framework path names used as link inputs in order to link against the dynamic
   * framework.
   */
  @SkylarkCallable(
      name = "framework_dirs",
      structField = true,
      doc =
          "The framework path names used as link inputs in order to link against the dynamic "
              + "framework.")
  Depset /*<String>*/ getDynamicFrameworkDirs();

  /**
   * Returns the full set of artifacts that should be included as inputs to link against the dynamic
   * framework.
   */
  @SkylarkCallable(
      name = "framework_files",
      structField = true,
      doc =
          "The full set of files that should be included as inputs to link against the "
              + "dynamic framework.")
  Depset /*<FileApiT>*/ getDynamicFrameworkFiles();

  /**
   * Returns the multi-architecture dylib binary of the dynamic framework. May return null if the
   * rule providing the framework only specified framework imports.
   */
  @Nullable
  @SkylarkCallable(
      name = "binary",
      allowReturnNones = true,
      structField = true,
      doc =
          "The multi-architecture dylib binary of the dynamic framework. May be None if "
              + "the rule providing the framework only specified framework imports.")
  FileApi getAppleDylibBinary();

  /**
   * Returns the {@link ObjcProviderApi} which contains information about the transitive
   * dependencies linked into the dylib.
   */
  @SkylarkCallable(
      name = "objc",
      structField = true,
      doc =
          "A provider which contains information about the transitive dependencies linked into "
              + "the dynamic framework.")
  ObjcProviderApi<FileApiT> getDepsObjcProvider();
}
