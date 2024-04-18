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

package com.google.devtools.build.lib.starlarkbuildapi.core;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Interface for the "struct" object in the build API. */
@StarlarkBuiltin(
    name = "struct",
    category = DocCategory.BUILTIN,
    doc =
        "A generic object with fields."
            + "<p>Structs fields cannot be reassigned once the struct is created. Two structs are "
            + "equal if they have the same fields and if corresponding field values are equal.")
public interface StructApi extends StarlarkValue {

  /** Callable Provider for new struct objects. */
  @StarlarkBuiltin(name = "Provider", documented = false, doc = "")
  interface StructProviderApi extends ProviderApi {

    @StarlarkMethod(
        name = "struct",
        doc =
            "Creates an immutable struct using the keyword arguments as attributes. It is used to "
                + "group multiple values together. Example:<br>"
                + "<pre class=\"language-python\">s = struct(x = 2, y = 3)\n"
                + "return s.x + getattr(s, \"y\")  # returns 5</pre>",
        extraKeywords =
            @Param(name = "kwargs", defaultValue = "{}", doc = "Dictionary of arguments."),
        useStarlarkThread = true,
        selfCall = true)
    @StarlarkConstructor
    StructApi createStruct(Dict<String, Object> kwargs, StarlarkThread thread) throws EvalException;
  }
}
