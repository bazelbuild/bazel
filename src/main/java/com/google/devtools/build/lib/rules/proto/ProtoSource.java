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

package com.google.devtools.build.lib.rules.proto;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Optional;

/** Represents a single {@code .proto} source file. */
@Immutable
@AutoCodec
class ProtoSource {
  private final Artifact sourceFile;
  private final Artifact originalSourceFile;
  private final PathFragment sourceRoot;
  private final Optional<PathFragment> importPath;

  public ProtoSource(
      Artifact sourceFile, PathFragment sourceRoot, Optional<PathFragment> importPath) {
    this(sourceFile, sourceFile, sourceRoot, importPath);
  }

  @AutoCodec.Instantiator
  ProtoSource(
      Artifact sourceFile,
      Artifact originalSourceFile,
      PathFragment sourceRoot,
      Optional<PathFragment> importPath) {
    this.sourceFile = sourceFile;
    this.originalSourceFile = originalSourceFile;
    this.sourceRoot = sourceRoot;
    this.importPath = importPath;
  }

  public Artifact getSourceFile() {
    return sourceFile;
  }

  /** Returns the original source file. Only for blacklisting protos! */
  @Deprecated
  Artifact getOriginalSourceFile() {
    return originalSourceFile;
  }

  public PathFragment getSourceRoot() {
    return sourceRoot;
  }

  public PathFragment getImportPath() {
    return importPath.orElseGet(() -> sourceFile.getExecPath().relativeTo(sourceRoot));
  }

  @Deprecated
  Optional<PathFragment> getImportPathForStrictImportableProtosImportPathsForDependents() {
    return importPath;
  }

  @Override
  public String toString() {
    return "ProtoSource('" + getImportPath() + "')";
  }
}
