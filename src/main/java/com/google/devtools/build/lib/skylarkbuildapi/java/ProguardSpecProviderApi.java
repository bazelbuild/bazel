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
package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/** Provides information about proguard specs for Android binaries. */
@SkylarkModule(
    name = "ProguardSpecProvider",
    doc = "Proguard specifications used for Android binaries.",
    category = SkylarkModuleCategory.PROVIDER)
public interface ProguardSpecProviderApi<FileT extends FileApi> extends StructApi {

  public static final String NAME = "ProguardSpecProvider";

  @SkylarkCallable(name = "specs", structField = true, doc = "", documented = false)
  NestedSet<FileT> getTransitiveProguardSpecs();

  /** The provider implementing this can construct the ProguardSpecProvider. */
  @SkylarkModule(name = "Provider", doc = "", documented = false)
  public interface Provider<FileT extends FileApi> extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>ProguardSpecProvider</code> constructor.",
        parameters = {
          @Param(
              name = "specs",
              doc = "Transitive proguard specs.",
              positional = true,
              named = false,
              type = SkylarkNestedSet.class,
              generic1 = FileApi.class),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = ProguardSpecProviderApi.class, receiverNameForDoc = NAME)
    ProguardSpecProviderApi<FileT> create(SkylarkNestedSet specs) throws EvalException;
  }
}
