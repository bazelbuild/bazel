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
package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/** Supplies the device broker type string, passed to the Android test runtime. */
@StarlarkBuiltin(
    name = "AndroidDeviceBrokerInfo",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed.",
    documented = false)
public interface AndroidDeviceBrokerInfoApi extends StructApi {

  /** Name of this info object. */
  String NAME = "AndroidDeviceBrokerInfo";

  /** Provider for {@link AndroidDeviceBrokerInfoApi}. */
  @StarlarkBuiltin(
      name = "Provider",
      doc =
          "Do not use this module. It is intended for migration purposes only. If you depend on "
              + "it, you will be broken when it is removed.",
      documented = false)
  interface AndroidDeviceBrokerInfoApiProvider extends ProviderApi {

    @StarlarkMethod(
        name = "AndroidDeviceBrokerInfo",
        doc = "The <code>AndroidDeviceBrokerInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "type",
              named = true,
              doc =
                  "The type of device broker that is appropriate to use to interact with "
                      + "devices")
        },
        selfCall = true)
    @StarlarkConstructor
    AndroidDeviceBrokerInfoApi createInfo(String type) throws EvalException;
  }
}
