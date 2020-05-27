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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** A target that provides C++ libraries to be linked into Python targets. */
@StarlarkBuiltin(
    name = "PyCcLinkParamsProvider",
    documented = false,
    category = StarlarkDocumentationCategory.PROVIDER,
    doc = "Wrapper for every C++ linking provider")
public interface PyCcLinkParamsProviderApi<FileT extends FileApi> extends StructApi {
  @StarlarkMethod(name = "cc_info", doc = "", structField = true, documented = false)
  CcInfoApi<FileT> getCcInfo();

  /** Provider for PyCcLinkParamsProvider objects. */
  @StarlarkBuiltin(name = "Provider", doc = "", documented = false)
  public interface Provider extends ProviderApi {}
}
