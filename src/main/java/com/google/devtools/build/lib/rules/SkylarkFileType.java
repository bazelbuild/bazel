// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.List;

/** A wrapper class for FileType and FileTypeSet functionality in Skylark. */
@SkylarkModule(
  name = "FileType",
  category = SkylarkModuleCategory.NONE,
  doc =
      "Deprecated. File type for file filtering. Can be used to filter collections of labels "
          + "for certain file types."
)
public class SkylarkFileType {

  private final FileType fileType;

  private SkylarkFileType(FileType fileType) {
    this.fileType = fileType;
  }

  public static SkylarkFileType of(Iterable<String> extensions) {
    return new SkylarkFileType(FileType.of(extensions));
  }

  public FileTypeSet getFileTypeSet() {
    return FileTypeSet.of(fileType);
  }

  @SkylarkCallable(doc =
      "Returns a list created from the elements of the parameter containing all the "
    + "<a href=\"File.html\"><code>File</code></a>s that match the FileType. The parameter "
    + "must be a <a href=\"set.html\"><code>set</code></a> or a "
    + "<a href=\"list.html\"><code>list</code></a>.")
  // toIterablesStrict() will ensure the parameter is a SkylarkNestedSet or a java Iterable
  // (including SkylarkList). If it fails, the error location information will be inserted by the
  // Skylark interface framework. If there's a dynamic type error on a non-Artifact element, the
  // error will also be handled by the Skylark interface framework.
  @SuppressWarnings("unchecked")
  public List<Artifact> filter(Object filesUnchecked) throws EvalException {
    return ImmutableList.copyOf(
        FileType.filter(
            (Iterable<Artifact>) EvalUtils.toIterableStrict(filesUnchecked, null),
            fileType));
  }

  @VisibleForTesting
  public Object getExtensions() {
    return fileType.getExtensions();
  }
}
