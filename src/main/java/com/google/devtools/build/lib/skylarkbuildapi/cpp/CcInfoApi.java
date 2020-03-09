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

import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.NoneType;

/** Wrapper for every C++ compilation and linking provider. */
@SkylarkModule(
    name = "CcInfo",
    category = SkylarkModuleCategory.PROVIDER,
    doc =
        "A provider for compilation and linking of C++. This "
            + "is also a marking provider telling C++ rules that they can depend on the rule "
            + "with this provider. If it is not intended for the rule to be depended on by C++, "
            + "the rule should wrap the CcInfo in some other provider.")
public interface CcInfoApi extends StructApi {
  String NAME = "CcInfo";

  @SkylarkCallable(
      name = "compilation_context",
      doc = "Returns the <code>CompilationContext</code>",
      structField = true)
  CcCompilationContextApi getCcCompilationContext();

  @SkylarkCallable(
      name = "linking_context",
      doc = "Returns the <code>LinkingContext</code>",
      structField = true)
  CcLinkingContextApi<?> getCcLinkingContext();

  /** The provider implementing this can construct CcInfo objects. */
  @SkylarkModule(
      name = "Provider",
      doc = "",
      // This object is documented via the CcInfo documentation and the docuemntation of its
      // callable function.
      documented = false)
  interface Provider extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>CcInfo</code> constructor.",
        parameters = {
          @Param(
              name = "compilation_context",
              doc = "The <code>CompilationContext</code>.",
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
              doc = "The <code>LinkingContext</code>.",
              positional = false,
              named = true,
              noneable = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = CcLinkingContextApi.class),
                @ParamType(type = NoneType.class)
              })
        },
        selfCall = true)
    @SkylarkConstructor(objectType = CcInfoApi.class, receiverNameForDoc = NAME)
    CcInfoApi createInfo(Object ccCompilationContext, Object ccLinkingInfo) throws EvalException;
  }
}
