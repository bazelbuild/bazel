// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi.python;

import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/**
 * DO NOT USE. Skarlark module exposing Python transitions for Python 2 to 3 migration purposes
 * only.
 */
@StarlarkBuiltin(
    name = "py_transitions",
    doc =
        "DO NOT USE. This is intended for Python 2 to 3 migration purposes only. If you depend"
            + " on it, you will be broken when it is removed.",
    documented = false)
public interface PyStarlarkTransitionsApi extends StarlarkValue {

  @StarlarkMethod(
      name = "cfg",
      doc =
          "DO NOT USE. This is intended for Python 2 to 3 migration purposes only. If you depend on"
              + " it, you will be broken when it is removed. A configuration that transitions to"
              + " the Python version specified by the 'python_version' attribute of the rule."
              + " Valid versions are: PY2, PY3, and DEFAULT. If 'python_version' attribute is not"
              + " available, or has an invalid value, it succeeds silently without transitions.",
      documented = false,
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API)
  public StarlarkValue getTransition();
}
