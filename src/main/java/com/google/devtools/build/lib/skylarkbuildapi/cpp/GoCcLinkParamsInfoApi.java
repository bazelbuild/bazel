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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;

/** A target that provides C++ libraries to be linked into Go targets. */
@SkylarkModule(
    name = "GoCcLinkParamsInfo",
    doc = "",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER)
public interface GoCcLinkParamsInfoApi extends StructApi {

  /** Provider for GoContextInfo objects. */
  @SkylarkModule(name = "Provider", doc = "", documented = false)
  public interface Provider<
          FileT extends FileApi, CcLinkingContextT extends CcLinkingContextApi<FileT>>
      extends ProviderApi {
    @SkylarkCallable(
        name = "GoCcLinkParamsInfo",
        doc = "The <code>GoCcLinkParamsInfo</code> constructor.",
        parameters = {
          @Param(
              name = "linking_context",
              doc = "The CC linking context.",
              positional = false,
              named = true,
              type = CcLinkingContextApi.class),
        },
        selfCall = true)
    @SkylarkConstructor(
        objectType = GoCcLinkParamsInfoApi.class,
        receiverNameForDoc = "GoCcLinkParamsInfo")
    public GoCcLinkParamsInfoApi createInfo(CcLinkingContextT ccLinkingContext)
        throws EvalException;
  }
}
