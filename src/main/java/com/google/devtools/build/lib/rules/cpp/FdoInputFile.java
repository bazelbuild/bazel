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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.util.FileType.HasFileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import net.starlark.java.eval.EvalException;

/** Value object reused by fdo configurations that may be either an artifact or a path. */
@Immutable
public final class FdoInputFile implements HasFileType {

  private final Artifact artifact;
  private final PathFragment absolutePath;

  private FdoInputFile(Artifact artifact, PathFragment absolutePath) {
    Preconditions.checkArgument((artifact == null) != (absolutePath == null));
    Preconditions.checkArgument(absolutePath == null || absolutePath.isAbsolute());
    this.artifact = artifact;
    this.absolutePath = absolutePath;
  }

  public Artifact getArtifact() {
    return artifact;
  }

  public PathFragment getAbsolutePath() {
    return absolutePath;
  }

  public String getBasename() {
    return artifact != null ? artifact.getFilename() : absolutePath.getBaseName();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }

    if (!(o instanceof FdoInputFile that)) {
      return false;
    }

    return Objects.equals(this.artifact, that.artifact)
        && Objects.equals(this.absolutePath, that.absolutePath);
  }

  @Override
  public int hashCode() {
    return Objects.hash(artifact, absolutePath);
  }

  public static FdoInputFile fromAbsolutePath(PathFragment absolutePath) {
    return new FdoInputFile(null, absolutePath);
  }

  public static FdoInputFile fromArtifact(Artifact artifact) {
    return new FdoInputFile(artifact, null);
  }

  public static FdoInputFile fromStarlarkProvider(StructImpl starlarkProvider)
      throws EvalException {
    String absolutePathStr = starlarkProvider.getValue("absolute_path", String.class);
    PathFragment absolutePath =
        absolutePathStr != null ? PathFragment.create(absolutePathStr) : null;
    return new FdoInputFile(starlarkProvider.getValue("artifact", Artifact.class), absolutePath);
  }

  @Override
  public String filePathForFileTypeMatcher() {
    return artifact != null
        ? artifact.filePathForFileTypeMatcher()
        : absolutePath.filePathForFileTypeMatcher();
  }
}
