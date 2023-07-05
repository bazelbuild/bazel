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
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.FileType.HasFileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.Nullable;

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

    if (!(o instanceof FdoInputFile)) {
      return false;
    }

    FdoInputFile that = (FdoInputFile) o;
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

  @Nullable
  public static FdoInputFile fromProfileRule(RuleContext ruleContext) throws InterruptedException {

    boolean isLabel = ruleContext.attributes().isAttributeValueExplicitlySpecified("profile");
    boolean isAbsolutePath =
        ruleContext.attributes().isAttributeValueExplicitlySpecified("absolute_path_profile");

    if (isLabel == isAbsolutePath) {
      ruleContext.ruleError("exactly one of profile and absolute_path_profile should be specified");
      return null;
    }

    if (isLabel) {
      Artifact artifact = ruleContext.getPrerequisiteArtifact("profile");
      return new FdoInputFile(artifact, null);
    } else {
      if (!ruleContext.getFragment(CppConfiguration.class).isFdoAbsolutePathEnabled()) {
        ruleContext.attributeError(
            "absolute_path_profile",
            "this attribute cannot be used when --enable_fdo_profile_absolute_path is false");
        return null;
      }
      String pathString = ruleContext.getExpander().expand("absolute_path_profile");
      PathFragment absolutePath = PathFragment.create(pathString);
      if (!absolutePath.isAbsolute()) {
        ruleContext.attributeError(
            "absolute_path_profile",
            String.format("%s is not an absolute path", absolutePath.getPathString()));
        return null;
      }
      return new FdoInputFile(null, absolutePath);
    }
  }

  @Override
  public String filePathForFileTypeMatcher() {
    return artifact != null
        ? artifact.filePathForFileTypeMatcher()
        : absolutePath.filePathForFileTypeMatcher();
  }
}
