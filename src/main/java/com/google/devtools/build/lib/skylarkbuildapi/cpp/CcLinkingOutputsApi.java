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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Interface for a structured representation of the linking outputs of a C++ rule. */
@SkylarkModule(
    name = "CcLinkingOutputs",
    category = SkylarkModuleCategory.BUILTIN,
    documented = false,
    doc = "Helper class containing CC compilation outputs.")
public interface CcLinkingOutputsApi<FileT extends FileApi> {
  @SkylarkCallable(name = "library_to_link", structField = true, documented = false)
  LibraryToLinkApi<FileT> getLibraryToLink();

  @SkylarkCallable(name = "executable", structField = true, documented = false)
  FileT getExecutable();
}
