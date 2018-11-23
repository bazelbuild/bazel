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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;

/** Wrapper for every C++ linking provider. */
@SkylarkModule(
    name = "cc_info",
    documented = false,
    category = SkylarkModuleCategory.PROVIDER,
    doc = "Wrapper for every C++ provider")
public interface CcInfoApi extends StructApi {
  String NAME = "CcInfo";

  @SkylarkCallable(
      name = "compilation_context",
      documented = false,
      allowReturnNones = true,
      structField = true)
  CcCompilationContextApi getCcCompilationContext();

  @SkylarkCallable(
      name = "linking_context",
      documented = false,
      allowReturnNones = true,
      structField = true)
  CcLinkingInfoApi getCcLinkingInfo();

  /** The provider implementing this can construct the AndroidCcLinkParamsInfo provider. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>CcInfo</code> constructor.",
        documented = false,
        useLocation = true,
        useEnvironment = true,
        parameters = {
          @Param(
              name = "compilation_context",
              doc = "The CcCompilationContext.",
              positional = false,
              named = true,
              noneable = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = CcCompilationContextApi.class),
                @ParamType(type = NoneType.class)
              }),
          @Param(
              name = "linking_context",
              doc = "The CcLinkingContext.",
              positional = false,
              named = true,
              noneable = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = CcLinkingContextApi.class),
                @ParamType(type = CcLinkingInfoApi.class),
                @ParamType(type = NoneType.class)
              })
        },
        selfCall = true)
    @SkylarkConstructor(objectType = CcInfoApi.class, receiverNameForDoc = NAME)
    CcInfoApi createInfo(
        Object ccCompilationContext,
        Object ccLinkingInfo,
        Location location,
        Environment environment)
        throws EvalException;
  }
}
