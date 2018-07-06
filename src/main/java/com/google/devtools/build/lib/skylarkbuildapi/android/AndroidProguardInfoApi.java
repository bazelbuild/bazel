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
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import java.util.List;

/** A target that can provide local proguard specifications. */
@SkylarkModule(name = "AndroidProguardInfo", doc = "", documented = false)
public interface AndroidProguardInfoApi<FileT extends FileApi> extends StructApi {
  String PROVIDER_NAME = "AndroidProguardInfo";

  @SkylarkCallable(
      name = "local_proguard_specs",
      structField = true,
      doc = "Returns the local proguard specs defined by this target.")
  ImmutableList<FileT> getLocalProguardSpecs();

  @SkylarkCallable(
      name = PROVIDER_NAME,
      doc = "The <code>AndroidProguardInfo</code> constructor.",
      parameters = {
        @Param(
            name = "local_proguard_specs",
            doc = "A list of local proguard specs.",
            positional = true,
            named = false,
            type = List.class
        )
      },
      selfCall = true)
  @SkylarkConstructor(objectType = AndroidProguardInfoApi.class)
  AndroidProguardInfoApi<FileT> androidProguardInfo(ImmutableList<FileT> localProguardSpecs);
}
