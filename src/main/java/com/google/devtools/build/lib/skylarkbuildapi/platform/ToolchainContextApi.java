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

package com.google.devtools.build.lib.skylarkbuildapi.platform;

import com.google.devtools.build.lib.syntax.StarlarkIndexable;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;

/** Stores toolchains available to a given rule. */
@StarlarkBuiltin(
    name = "ToolchainContext",
    category = StarlarkDocumentationCategory.BUILTIN,
    doc =
        "Holds toolchains available for a particular exec group. Toolchain targets are accessed by"
            + " indexing with the toolchain type, as in"
            + " <code>context[\"//pkg:my_toolchain_type\"]</code>.")
public interface ToolchainContextApi extends StarlarkValue, StarlarkIndexable {}
