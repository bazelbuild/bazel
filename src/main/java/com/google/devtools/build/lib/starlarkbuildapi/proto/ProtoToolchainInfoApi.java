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

package com.google.devtools.build.lib.starlarkbuildapi.proto;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/** Information about the {@code proto} toolchain. */
@StarlarkBuiltin(
    name = ProtoToolchainInfoApi.NAME,
    category = DocCategory.PROVIDER,
    doc = "Information about the `proto` toolchain.")
public interface ProtoToolchainInfoApi<
        FilesToRunProviderApiT extends FilesToRunProviderApi<? extends FileApi>>
    extends StructApi {
  /** The name of the provider in Starlark. */
  String NAME = "ProtoToolchainInfo";

  @StarlarkMethod(name = "compiler", doc = "The proto compiler to use.", structField = true)
  FilesToRunProviderApiT getCompiler();

  @StarlarkMethod(
      name = "compiler_options",
      doc = "Additional options to pass to `protoc`.",
      structField = true)
  ImmutableList<String> getCompilerOptions();

  /** The provider implementing this can construct {@code ProtoToolchainInfo} objects. */
  @StarlarkBuiltin(
      name = "Provider",
      doc = "",
      // This object is documented via the ProtoToolchainInfo documentation and the docuemntation of
      // its
      // callable function.
      documented = false)
  interface Provider<FilesToRunProviderApiT extends FilesToRunProviderApi<? extends FileApi>>
      extends ProviderApi {
    @StarlarkMethod(
        name = NAME,
        doc = "The `ProtoToolchainInfo` constructor.",
        parameters = {
          @Param(
              name = "compiler",
              doc = "The proto compiler.",
              positional = false,
              named = true,
              allowedTypes = {@ParamType(type = FilesToRunProviderApi.class)}),
          @Param(
              name = "compiler_options",
              doc = "The proto compiler.",
              positional = false,
              named = true,
              defaultValue = "[]",
              allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)})
        },
        selfCall = true)
    @StarlarkConstructor
    ProtoToolchainInfoApi<FilesToRunProviderApiT> create(
        FilesToRunProviderApiT protoc, Sequence<?> protocOptions) throws EvalException;
  }
}
