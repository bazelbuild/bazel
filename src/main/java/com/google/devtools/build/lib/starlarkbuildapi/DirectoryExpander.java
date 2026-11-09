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
import com.google.devtools.build.docgen.annot.DocCategory;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

/** A class that can be used to expand directories at execution time. */
@StarlarkBuiltin(
    name = "DirectoryExpander",
    category = DocCategory.BUILTIN,
    doc =
        "Expands directories created by <a href='../builtins/actions.html#declare_directory'>"
            + "<code>ctx.actions.declare_directory</code></a>"
            + " during the execution phase. This is useful to expand directories in "
            + "<a href='../builtins/Args.html#add_all.map_each'><code>map_each</code></a>.")
public interface DirectoryExpander extends StarlarkValue {
  @StarlarkMethod(
      name = "expand",
      doc =
          "If the given <code>File</code> is a declared directory output, this returns its"
              + " descendant files recursively. Otherwise, this returns a list containing just"
              + " the given <code>File</code>.",
      parameters = {
        @Param(
            name = "file",
            positional = true,
            named = false,
            doc = "The directory or file to expand."),
      })
  ImmutableList<FileApi> list(FileApi artifact) throws EvalException;

  @StarlarkMethod(
      name = "digest",
      allowReturnNones = true,
      doc =
          "Returns an opaque content digest for a declared regular file input of the action,"
              + " including a file in an expanded directory. Use the digest to compare file"
              + " contents within the action. For a declared source directory, this returns"
              + " a digest of the tracked paths and contents when source-directory tracking"
              + " is enabled. The digest's format and hash algorithm may change."
              + " If input metadata is unavailable, for example in <code>aquery</code>, this"
              + " returns <code>&lt;DIGEST&gt;</code> by default. Actions that discover inputs,"
              + " including actions with <code>unused_inputs_list</code>, cannot read digests.",
      parameters = {
        @Param(name = "file", positional = true, named = false, doc = "The input file."),
        @Param(
            name = "allow_missing",
            positional = false,
            named = true,
            defaultValue = "False",
            doc = "Return <code>None</code> when no content digest is available."),
      })
  @Nullable
  String digest(FileApi file, boolean allowMissing) throws EvalException;
}
