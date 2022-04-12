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

package com.google.devtools.build.lib.starlarkbuildapi.config;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Starlark API for accessing details of a ToolchainTypeRequirement. */
@StarlarkBuiltin(
    name = "toolchain_type",
    category = DocCategory.BUILTIN,
    doc = "A data type describing a dependency on a specific toolchain type.")
public interface StarlarkToolchainTypeRequirement extends StarlarkValue {

  @StarlarkMethod(
      name = "toolchain_type",
      doc = "The toolchain type that is required.",
      structField = true)
  Label toolchainType();

  @StarlarkMethod(
      name = "mandatory",
      doc = "Whether the toolchain type is mandatory or optional.",
      structField = true)
  boolean mandatory();
}
