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

package com.google.devtools.build.lib.starlarkbuildapi.go;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;

/** Contains the metadata for a Go package. Used to generate .gopackage files. */
@StarlarkBuiltin(
    name = "GoPackageInfo",
    doc = "",
    documented = false,
    category = DocCategory.PROVIDER)
public interface GoPackageInfoApi extends StructApi {
  String PROVIDER_NAME = "GoPackageInfo";

  /** Provider for GoPackageInfo objects. */
  @StarlarkBuiltin(name = "Provider", doc = "", documented = false)
  public interface Provider extends ProviderApi {
    @StarlarkMethod(
        name = PROVIDER_NAME,
        documented = false,
        parameters = {
          @Param(name = "label", positional = false, named = true),
          @Param(name = "srcs", positional = false, named = true),
          @Param(
              name = "export_data",
              positional = false,
              named = true,
              allowedTypes = {@ParamType(type = FileApi.class), @ParamType(type = NoneType.class)}),
          @Param(name = "imports", positional = false, named = true),
          @Param(
              name = "library",
              positional = false,
              named = true,
              allowedTypes = {
                @ParamType(type = GoPackageInfoApi.class),
                @ParamType(type = NoneType.class)
              },
              defaultValue = "None"),
          @Param(name = "test", positional = false, named = true),
          @Param(
              name = "is_proto_library",
              positional = false,
              named = true,
              defaultValue = "False"),
        },
        selfCall = true)
    @StarlarkConstructor
    GoPackageInfoApi createGoPackageInfo(
        Label nonProtoLibraryLabel,
        Sequence<?> srcs,
        Object exportDataObject,
        Sequence<?> imports,
        Object library,
        boolean test,
        boolean isProtolibrary)
        throws EvalException;
  }
}
