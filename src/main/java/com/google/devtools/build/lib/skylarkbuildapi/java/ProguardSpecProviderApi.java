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

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.syntax.EvalException;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkConstructor;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Provides information about proguard specs for Android binaries. */
@StarlarkBuiltin(
    name = "ProguardSpecProvider",
    doc = "Proguard specifications used for Android binaries.",
    category = StarlarkDocumentationCategory.PROVIDER)
public interface ProguardSpecProviderApi<FileT extends FileApi> extends StructApi {

  String NAME = "ProguardSpecProvider";

  @StarlarkMethod(name = "specs", structField = true, doc = "", documented = false)
  Depset /*<FileT>*/ getTransitiveProguardSpecsForStarlark();

  /** The provider implementing this can construct the ProguardSpecProvider. */
  @StarlarkBuiltin(name = "Provider", doc = "", documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>ProguardSpecProvider</code> constructor.",
        parameters = {
          @Param(
              name = "specs",
              doc = "Transitive proguard specs.",
              positional = true,
              named = false,
              type = Depset.class,
              generic1 = FileApi.class),
        },
        selfCall = true)
    @StarlarkConstructor(objectType = ProguardSpecProviderApi.class, receiverNameForDoc = NAME)
    ProguardSpecProviderApi<FileT> create(Depset specs) throws EvalException;
  }
}
