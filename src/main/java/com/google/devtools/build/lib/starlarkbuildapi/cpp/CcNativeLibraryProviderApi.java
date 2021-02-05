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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;

/**
 * Provider for native libraries in the transitive closure of a target that are needed for executing
 * C++ code.
 */
@StarlarkBuiltin(name = "CcNativeLibrary", documented = false, category = DocCategory.PROVIDER)
public interface CcNativeLibraryProviderApi extends StructApi {

  @StarlarkMethod(name = "libs", structField = true, documented = false)
  Depset getTransitiveCcNativeLibrariesStarlark() throws EvalException;
}
