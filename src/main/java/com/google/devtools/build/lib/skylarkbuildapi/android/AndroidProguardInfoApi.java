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
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;

/** A target that can provide local proguard specifications. */
@SkylarkModule(
    name = "AndroidProguardInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidProguardInfoApi<FileT extends FileApi> extends StructApi {
  /** The name of the provider for this info object. */
  String NAME = "AndroidProguardInfo";

  @SkylarkCallable(
      name = "local_proguard_specs",
      structField = true,
      doc = "Returns the local proguard specs defined by this target.",
      documented = false)
  ImmutableList<FileT> getLocalProguardSpecs();

  /** The provider implementing this can construct the AndroidProguardInfo provider. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>AndroidProguardInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "local_proguard_specs",
              doc = "A list of local proguard specs.",
              positional = true,
              named = false,
              type = Sequence.class,
              generic1 = FileApi.class)
        },
        selfCall = true)
    @SkylarkConstructor(objectType = AndroidProguardInfoApi.class)
    AndroidProguardInfoApi<FileT> createInfo(Sequence<?> localProguardSpecs /* <FileT> */)
        throws EvalException;
  }
}
