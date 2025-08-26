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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;

/** Wrapper for every C++ linking provider. */
@StarlarkBuiltin(
    name = "LinkingContext",
    category = DocCategory.BUILTIN,
    doc =
        "Immutable store of information needed for C++ linking that is aggregated across "
            + "dependencies.")
public interface CcLinkingContextApi extends StarlarkValue {
  @StarlarkMethod(
      name = "linker_inputs",
      doc = "Returns the depset of linker inputs.",
      structField = true)
  public default Depset getStarlarkLinkerInputs() {
    throw new UnsupportedOperationException();
  }

  @StarlarkMethod(name = "extra_link_time_libraries", documented = false, useStarlarkThread = true)
  public default Object getExtraLinkTimeLibrariesForStarlark(StarlarkThread thread)
      throws EvalException {
    throw new UnsupportedOperationException();
  }
}
