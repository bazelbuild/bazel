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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppSemantics;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Semantics for Bazel Objc rules. */
@StarlarkBuiltin(name = "bazel_objc_internal", category = DocCategory.BUILTIN, documented = false)
public class BazelObjcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "bazel_objc_internal";

  @StarlarkMethod(name = "semantics", documented = false, structField = true)
  public CppSemantics getSemantics() {
    return BazelCppSemantics.OBJC;
  }
}
