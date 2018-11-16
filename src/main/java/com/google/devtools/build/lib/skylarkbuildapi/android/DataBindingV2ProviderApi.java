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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * An interface for a provider that exposes the use of <a
 * href="https://developer.android.com/topic/libraries/data-binding/index.html">data binding</a>.
 */
@SkylarkModule(
    name = "DataBindingV2Info",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface DataBindingV2ProviderApi<T extends FileApi> extends StructApi {

  /** Name of this info object. */
  public static final String NAME = "DataBindingV2Info";

  /** Returns the setter store files from this rule. */
  @SkylarkCallable(name = "setter_stores", structField = true, doc = "", documented = false)
  ImmutableList<T> getSetterStores();

  /** Returns the client info files from this rule. */
  @SkylarkCallable(name = "client_infos", structField = true, doc = "", documented = false)
  ImmutableList<T> getClassInfos();

  /** Returns the BR files from this rule and its dependencies. */
  @SkylarkCallable(name = "transitive_br_files", structField = true, doc = "", documented = false)
  NestedSet<T> getTransitiveBRFiles();

  /** The provider implementing this can construct the DataBindingV2Info provider. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  public interface Provider<F extends FileApi> extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>DataBindingV2Info</code> constructor.",
        documented = false,
        parameters = {
            @Param(
                name = "setter_stores",
                doc = "A list of artifacts of setter_stores.bin.",
                positional = true,
                named = false,
                type = SkylarkList.class,
                generic1 = FileApi.class),
            @Param(
                name = "client_infos",
                doc = "A list of artifacts of client_infos.bin.",
                positional = true,
                named = false,
                type = SkylarkList.class,
                generic1 = FileApi.class),
            @Param(
                name = "transitive_br_files",
                doc = "A list of artifacts of br.bin.",
                positional = true,
                named = false,
                type = SkylarkNestedSet.class,
                generic1 = FileApi.class),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = DataBindingV2ProviderApi.class)
    DataBindingV2ProviderApi<F> createInfo(
        SkylarkList<F> setterStores,
        SkylarkList<F> clientInfos,
        SkylarkNestedSet transitiveBrFiles) throws EvalException;
  }
}