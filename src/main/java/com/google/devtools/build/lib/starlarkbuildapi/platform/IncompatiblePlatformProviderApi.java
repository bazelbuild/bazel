// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.platform;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/** Targets that are incompatible with the target platform. */
@StarlarkBuiltin(
    name = "IncompatiblePlatformProvider",
    doc =
        "A provider for targets that are incompatible with the target platform. See "
            + "<a href='${link platforms#detecting-incompatible-targets-using-bazel-cquery}'>"
            + "Detecting incompatible targets using <code>bazel cquery</code></a> for more "
            + "information."
            + "<p>Returning an instance of this provider from a rule's implementation function "
            + "marks the target as incompatible with the current platform, in the same way that "
            + "the <code>target_compatible_with</code> attribute does. This allows rules to make "
            + "incompatibility decisions based on dynamic logic that can't easily be expressed "
            + "via constraints.</p>",
    category = DocCategory.PROVIDER)
public interface IncompatiblePlatformProviderApi extends StarlarkValue {

  /** Provider for {@link IncompatiblePlatformProviderApi}. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface IncompatiblePlatformProviderApiProvider extends ProviderApi {

    @StarlarkMethod(
        name = "IncompatiblePlatformProvider",
        doc =
            "Constructs an <code>IncompatiblePlatformProvider</code>. Returning this provider "
                + "from a rule's implementation function marks the target as incompatible with "
                + "the current platform.",
        parameters = {
          @Param(
              name = "constraints",
              allowedTypes = {
                @ParamType(type = Sequence.class, generic1 = ConstraintValueInfoApi.class),
                @ParamType(type = NoneType.class),
              },
              named = true,
              positional = false,
              defaultValue = "None",
              doc =
                  "An optional list of <a href='ConstraintValueInfo.html'>"
                      + "<code>ConstraintValueInfo</code></a> instances that the target platform "
                      + "did not satisfy, used for explanatory error messages when an incompatible "
                      + "target is explicitly requested on the command line."),
        },
        selfCall = true)
    @StarlarkConstructor
    IncompatiblePlatformProviderApi constructor(Object constraints) throws EvalException;
  }
}
