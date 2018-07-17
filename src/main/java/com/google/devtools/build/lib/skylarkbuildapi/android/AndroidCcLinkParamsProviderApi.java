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

import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkParamsStoreApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;

/** A target that provides C++ libraries to be linked into Android targets. */
@SkylarkModule(
    name = "AndroidCcLinkParamsInfo",
    doc = "Information about the c++ libraries to be linked into Android targets.",
    category = SkylarkModuleCategory.PROVIDER)
public interface AndroidCcLinkParamsProviderApi<T extends CcLinkParamsStoreApi> extends StructApi {
  /** Name of this info object. */
  public static String NAME = "AndroidCcLinkParamsInfo";

  /** Returns the cc link params. */
  @SkylarkCallable(name = "link_params", structField = true, doc = "", documented = false)
  T getLinkParams();

  /** The provider implementing this can construct the AndroidCcLinkParamsInfo provider. */
  @SkylarkModule(name = "Provider", doc = "", documented = false)
  public interface Provider<T extends CcLinkParamsStoreApi> extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>AndroidCcLinkParamsInfo</code> constructor.",
        parameters = {
          @Param(
              name = "store",
              doc = "The cc link params store.",
              positional = true,
              named = false,
              type = CcLinkParamsStoreApi.class),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = AndroidCcLinkParamsProviderApi.class)
    public AndroidCcLinkParamsProviderApi<T> createInfo(T store) throws EvalException;
  }
}
