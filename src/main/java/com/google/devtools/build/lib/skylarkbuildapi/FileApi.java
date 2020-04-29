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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** The interface for files in Starlark. */
@SkylarkModule(
    name = "File",
    category = SkylarkModuleCategory.BUILTIN,
    doc =
        "This object is created during the analysis phase to represent a file or directory that "
            + "will be read or written during the execution phase. It is not an open file"
            + " handle, "
            + "and cannot be used to directly read or write file contents. Rather, you use it to "
            + "construct the action graph in a rule implementation function by passing it to "
            + "action-creating functions. See the "
            + "<a href='../rules.$DOC_EXT#files'>Rules page</a> for more information."
            + "" // curse google-java-format b/145078219
            + "<p>When a <code>File</code> is passed to an <a"
            + " href='Args.html'><code>Args</code></a> object without using a"
            + " <code>map_each</code> function, it is converted to a string by taking the value of"
            + " its <code>path</code> field.")
public interface FileApi extends StarlarkValue {

  @SkylarkCallable(
      name = "dirname",
      structField = true,
      doc =
          "The name of the directory containing this file. It's taken from "
              + "<a href=\"#path\">path</a> and is always relative to the execution directory.")
  String getDirname();

  @SkylarkCallable(
      name = "basename",
      structField = true,
      doc = "The base name of this file. This is the name of the file inside the directory.")
  String getFilename();

  @SkylarkCallable(name = "extension", structField = true, doc = "The file extension of this file.")
  String getExtension();

  @SkylarkCallable(
      name = "owner",
      structField = true,
      allowReturnNones = true,
      doc = "A label of a target that produces this File.")
  Label getOwnerLabel();

  @SkylarkCallable(
      name = "root",
      structField = true,
      doc = "The root beneath which this file resides.")
  FileRootApi getRoot();

  @SkylarkCallable(
      name = "is_source",
      structField = true,
      doc = "Returns true if this is a source file, i.e. it is not generated.")
  boolean isSourceArtifact();

  // TODO(rduan): Document this Starlark method once TreeArtifact is no longer experimental.
  @SkylarkCallable(name = "is_directory", structField = true, documented = false)
  boolean isDirectory();

  @SkylarkCallable(
      name = "short_path",
      structField = true,
      doc =
          "The path of this file relative to its root. This excludes the aforementioned "
              + "<i>root</i>, i.e. configuration-specific fragments of the path. This is also the "
              + "path under which the file is mapped if it's in the runfiles of a binary.")
  String getRunfilesPathString();

  @SkylarkCallable(
      name = "path",
      structField = true,
      doc =
          "The execution path of this file, relative to the workspace's execution directory. It "
              + "consists of two parts, an optional first part called the <i>root</i> (see also "
              + "the <a href=\"root.html\">root</a> module), and the second part which is the "
              + "<code>short_path</code>. The root may be empty, which it usually is for "
              + "non-generated files. For generated files it usually contains a "
              + "configuration-specific path fragment that encodes things like the target CPU "
              + "architecture that was used while building said file. Use the "
              + "<code>short_path</code> for the path under which the file is mapped if it's in "
              + "the runfiles of a binary.")
  String getExecPathString();
}
