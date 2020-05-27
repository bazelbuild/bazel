// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkIndexable;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Stores {@link com.google.devtools.build.lib.packages.ExecGroup}s available to a given rule. */
@StarlarkBuiltin(
    name = "ExecGroupCollection",
    category = StarlarkDocumentationCategory.BUILTIN,
    // TODO(b/151742236) update this doc when this becomes non-experimental.
    doc = "<i>experimental</i> Stores exec groups available to a given rule.")
public interface ExecGroupCollectionApi extends StarlarkValue, StarlarkIndexable {

  /**
   * Stores information about a single ExecGroup. The StarlarkMethod functions in this module should
   * be a subset of the StarlarkMethod functions available for the default exec group via {@link
   * com.google.devtools.build.lib.skylarkbuildapi.StarlarkRuleContextApi}. This allows a user to
   * pass in a rule ctx to the same places that take an exec group ctx to have them operate on the
   * default exec group.
   */
  @StarlarkBuiltin(
      name = "ExecGroupContext",
      category = StarlarkDocumentationCategory.BUILTIN,
      // TODO(b/151742236) update this doc when this becomes non-experimental.
      doc = "<i>experimental</i> Stores information about an exec group.")
  interface ExecGroupContextApi extends StarlarkValue {
    @StarlarkMethod(
        name = "toolchains",
        structField = true,
        // TODO(b/151742236) update this doc when this becomes non-experimental.
        doc = "<i>experimental</i> Toolchains required for this exec group")
    ToolchainContextApi toolchains() throws EvalException;
  }
}
