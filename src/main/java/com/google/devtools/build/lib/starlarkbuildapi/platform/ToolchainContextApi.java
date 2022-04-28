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

package com.google.devtools.build.lib.starlarkbuildapi.platform;

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkIndexable;
import net.starlark.java.eval.StarlarkValue;

/** Stores toolchains available to a given rule. */
@StarlarkBuiltin(
    name = "ToolchainContext",
    category = DocCategory.BUILTIN,
    doc =
        "Holds toolchains available for a particular exec group. Toolchain targets are accessed by"
            + " indexing with the toolchain type, as in"
            + " <code>ctx.toolchains[\"//pkg:my_toolchain_type\"]</code>. If the toolchain was"
            + " optional and no toolchain was resolved, this will return <code>None</code>.")
public interface ToolchainContextApi extends StarlarkValue, StarlarkIndexable.Threaded {}
