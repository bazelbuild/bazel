// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi.proto;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkDocumentationCategory;
import com.google.devtools.build.lib.syntax.Sequence;

/** Provides information about flavors for all built protos. */
@StarlarkBuiltin(
    name = "ProtoRegistryProvider",
    doc = "Information about flavors for all built protos.",
    category = StarlarkDocumentationCategory.PROVIDER)
public interface ProtoRegistryProviderApi<FileT extends FileApi> extends StructApi {

  @SkylarkCallable(name = "jars", documented = false, doc = "", structField = true)
  Depset /*<FileT>*/ getJars();

  @SkylarkCallable(name = "flavors", documented = false, doc = "", structField = true)
  Sequence<String> getFlavors();

  @SkylarkCallable(
      name = "errorMessage",
      documented = false,
      doc = "",
      structField = true,
      allowReturnNones = true)
  String getError();
}
