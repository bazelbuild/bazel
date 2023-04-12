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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/** The interface for Starlark-defined aspects in the Build API. */
@StarlarkBuiltin(
    name = "Aspect",
    category = DocCategory.BUILTIN,
    doc =
        "For more information about Aspects, please consult the <a"
            + " href=\"../globals/bzl.html#aspect\">documentation of the aspect function</a> or the"
            + " <a href=\"https://bazel.build/rules/aspects\">introduction to Aspects</a>.")
public interface StarlarkAspectApi extends StarlarkValue {}
