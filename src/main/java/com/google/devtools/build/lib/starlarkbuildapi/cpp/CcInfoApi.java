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
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.StarlarkThread;

/** Wrapper for every C++ compilation and linking provider. */
@StarlarkBuiltin(
    name = "CcInfo",
    category = DocCategory.PROVIDER,
    doc =
        "A provider for compilation and linking of C++. This "
            + "is also a marking provider telling C++ rules that they can depend on the rule "
            + "with this provider. If it is not intended for the rule to be depended on by C++, "
            + "the rule should wrap the CcInfo in some other provider.")
public interface CcInfoApi<FileT extends FileApi> extends StructApi {
  String NAME = "CcInfo";

  @StarlarkMethod(
      name = "compilation_context",
      doc = "Returns the <code>CompilationContext</code>",
      structField = true)
  CcCompilationContextApi<FileT> getCcCompilationContext();

  @StarlarkMethod(
      name = "linking_context",
      doc = "Returns the <code>LinkingContext</code>",
      structField = true)
  CcLinkingContextApi<?> getCcLinkingContext();

  @StarlarkMethod(
      name = "get_debug_context",
      documented = false,
      doc = "Returns the <code>DebugContext</code>",
      useStarlarkThread = true)
  CcDebugInfoContextApi getCcDebugInfoContextFromStarlark(StarlarkThread thread)
      throws EvalException;

  /** The provider implementing this can construct CcInfo objects. */
  @StarlarkBuiltin(
      name = "Provider",
      doc = "",
      // This object is documented via the CcInfo documentation and the docuemntation of its
      // callable function.
      documented = false)
  interface Provider<FileT extends FileApi> extends ProviderApi {

    @StarlarkMethod(
        name = NAME,
        doc = "The <code>CcInfo</code> constructor.",
        parameters = {
          @Param(
              name = "compilation_context",
              doc = "The <code>CompilationContext</code>.",
              positional = false,
              named = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = CcCompilationContextApi.class),
                @ParamType(type = NoneType.class)
              }),
          @Param(
              name = "linking_context",
              doc = "The <code>LinkingContext</code>.",
              positional = false,
              named = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = CcLinkingContextApi.class),
                @ParamType(type = NoneType.class)
              }),
          @Param(
              name = "debug_context",
              doc = "The <code>DebugContext</code>.",
              positional = false,
              named = true,
              defaultValue = "None",
              allowedTypes = {
                @ParamType(type = CcDebugInfoContextApi.class),
                @ParamType(type = NoneType.class)
              })
        },
        selfCall = true)
    @StarlarkConstructor
    CcInfoApi<FileT> createInfo(
        Object ccCompilationContext, Object ccLinkingInfo, Object ccDebugInfo) throws EvalException;
  }
}
