// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.syntax.EvalException;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkConstructor;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/**
 * A provider for targets that create Android instrumentations. Consumed by Android testing rules.
 */
@StarlarkBuiltin(
    name = "AndroidInstrumentationInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Android instrumentation and target APKs to run in a test",
    documented = false,
    category = StarlarkDocumentationCategory.PROVIDER)
public interface AndroidInstrumentationInfoApi<ApkT extends ApkInfoApi<?>> extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidInstrumentationInfo";

  @StarlarkMethod(
      name = "target",
      doc = "Returns the target ApkInfo of the instrumentation test.",
      documented = false,
      structField = true,
      allowReturnNones = true)
  ApkT getTarget();

  /** Provider for {@link AndroidInstrumentationInfoApi}. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface AndroidInstrumentationInfoApiProvider<ApkT extends ApkInfoApi<?>> extends ProviderApi {

    @StarlarkMethod(
        name = "AndroidInstrumentationInfo",
        doc = "The <code>AndroidInstrumentationInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "target",
              type = ApkInfoApi.class,
              named = true,
              doc = "The target ApkInfo of the instrumentation test.")
        },
        selfCall = true)
    @StarlarkConstructor(
        objectType = AndroidInstrumentationInfoApi.class,
        receiverNameForDoc = NAME)
    AndroidInstrumentationInfoApi<ApkT> createInfo(ApkT target) throws EvalException;
  }
}
