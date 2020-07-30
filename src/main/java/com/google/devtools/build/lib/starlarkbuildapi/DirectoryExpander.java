// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** A class that can be used to expand directories at execution time. */
@StarlarkBuiltin(
    name = "DirectoryExpander",
    category = StarlarkDocumentationCategory.BUILTIN,
    doc =
        "Expands directories created by <a href='actions.html#declare_directory'>"
            + "<code>ctx.actions.declare_directory</code></a>"
            + " during the execution phase. This is useful to expand directories in "
            + "<a href='Args.html#add_all.map_each'><code>map_each</code></a>.")
public interface DirectoryExpander extends StarlarkValue {
  @StarlarkMethod(
      name = "expand",
      doc =
          "If the given <code>File</code> is a directory, this returns a list of <code>File"
              + "</code>s recursively underneath the directory. Otherwise, this returns a list "
              + "containing just the given <code>File</code> itself.",
      parameters = {
        @Param(
            name = "file",
            type = FileApi.class,
            positional = true,
            named = false,
            doc = "The directory or file to expand."),
      })
  ImmutableList<FileApi> list(FileApi artifact);
}
