// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Bridge to allow builtins bzl code to call Java code. */
@StarlarkBuiltin(name = "py_builtins", documented = false)
public abstract class PyBuiltins implements StarlarkValue {
  public static final String NAME = "py_builtins";

  @StarlarkMethod(
      name = "is_singleton_depset",
      doc = "Efficiently checks if the depset is a singleton.",
      parameters = {
        @Param(
            name = "value",
            positional = true,
            named = false,
            defaultValue = "unbound",
            doc = "depset to check for being a singleton")
      })
  public boolean isSingletonDepset(Depset depset) {
    return depset.getSet().isSingleton();
  }
}
