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
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Additional layer of configurability for c++ rules through features and actions. */
@StarlarkBuiltin(
    name = "CcToolchainConfigInfo",
    category = StarlarkDocumentationCategory.PROVIDER,
    doc =
        "Additional layer of configurability for C++ rules. Encapsulates platform-dependent "
            + "specifics of C++ actions through features and action configs. It is used to "
            + "configure the C++ toolchain, and later on for command line construction. "
            + "Replaces the functionality of CROSSTOOL file.")
public interface CcToolchainConfigInfoApi extends StructApi {
  @StarlarkMethod(
      name = "proto",
      doc = "Returns CToolchain text proto from the CcToolchainConfigInfo data.",
      structField = true)
  String getProto();

  /** Provider class for {@link CcToolchainConfigInfoApi} objects. */
  @StarlarkBuiltin(
      name = "Provider",
      // This object is documented via the CcInfo documentation and the docuemntation of its
      // callable function.
      documented = false,
      doc = "")
  public interface Provider extends ProviderApi {}
}
