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
package com.google.devtools.build.lib.analysis.skylark;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.FileTypeApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;

/** A wrapper class for FileType and FileTypeSet functionality in Skylark. */
@AutoCodec
public class SkylarkFileType implements FileTypeApi<Artifact> {

  private final FileType fileType;

  @AutoCodec.VisibleForSerialization
  SkylarkFileType(FileType fileType) {
    this.fileType = fileType;
  }

  public static SkylarkFileType of(List<String> extensions) {
    return new SkylarkFileType(FileType.of(extensions));
  }

  public FileTypeSet getFileTypeSet() {
    return FileTypeSet.of(fileType);
  }

  @Override
  public ImmutableList<Artifact> filter(Object filesUnchecked) throws EvalException {
    return ImmutableList.copyOf(
        FileType.filter(
            (Iterable<Artifact>) EvalUtils.toIterableStrict(filesUnchecked, null, null), fileType));
  }

  @VisibleForTesting
  public Object getExtensions() {
    return fileType.getExtensions();
  }

  @Override
  public int hashCode() {
    return fileType.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    return other == this
        || (other instanceof SkylarkFileType
            && this.fileType.equals(((SkylarkFileType) other).fileType));
  }

  @Override
  public String toString() {
    return fileType.toString();
  }
}
