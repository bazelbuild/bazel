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

import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** A target that provides C++ libraries to be linked into Java targets. */
@SkylarkModule(
    name = "JavaCcLinkParamsInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Information about the c++ libraries to be linked into Java targets.",
    documented = true,
    category = SkylarkModuleCategory.PROVIDER)
public interface JavaCcLinkParamsProviderApi<CcInfoApiT extends CcInfoApi> extends StarlarkValue {
  /** Name of this info object. */
  String NAME = "JavaCcLinkParamsInfo";

  /** Returns the cc linking info */
  @SkylarkCallable(
      name = "cc_info",
      structField = true,
      doc = "Returns the CcLinkingInfo provider.",
      documented = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS)
  CcInfoApiT getCcInfo();

  /** The provider implementing this can construct the JavaCcLinkParamsInfo provider. */
  @SkylarkModule(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface Provider<CcInfoApiT extends CcInfoApi> extends ProviderApi {

    @SkylarkCallable(
        name = NAME,
        doc = "The <code>JavaCcLinkParamsInfo</code> constructor.",
        documented = true,
        enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS,
        parameters = {
          @Param(
              name = "store",
              doc = "The CcInfo provider.",
              positional = true,
              named = false,
              type = CcInfoApi.class),
        },
        selfCall = true)
    @SkylarkConstructor(objectType = JavaCcLinkParamsProviderApi.class, receiverNameForDoc = NAME)
    JavaCcLinkParamsProviderApi<CcInfoApiT> createInfo(CcInfoApiT store) throws EvalException;
  }
}
