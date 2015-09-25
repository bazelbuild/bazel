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

package com.google.devtools.build.xcode.xcodegen;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.xcode.util.Value;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;

import java.nio.file.FileSystem;
import java.nio.file.Path;

/**
 * Source file path with information on how to build it.
 */
public class SourceFile extends Value<SourceFile> {
  /** Indicates how a source file is built or not built. */
  public enum BuildType {
    NO_BUILD, BUILD, NON_ARC_BUILD;
  }

  private final BuildType buildType;
  private final Path path;

  private SourceFile(BuildType buildType, Path path) {
    super(buildType, path);
    this.buildType = buildType;
    this.path = path;
  }

  public BuildType buildType() {
    return buildType;
  }

  public Path path() {
    return path;
  }

  /**
   * Returns information on all source files in a target. In particular, this includes:
   * <ul>
   *   <li>arc-compiled source files
   *   <li>non-arc-compiled source files
   *   <li>support files, such as BUILD and header files
   *   <li>Info.plist file
   * </ul>
   */
  public static Iterable<SourceFile> allSourceFiles(FileSystem fileSystem, TargetControl control) {
    ImmutableList.Builder<SourceFile> result = new ImmutableList.Builder<>();
    for (Path plainSource : RelativePaths.fromStrings(fileSystem, control.getSourceFileList())) {
      result.add(new SourceFile(BuildType.BUILD, plainSource));
    }
    for (Path nonArcSource
        : RelativePaths.fromStrings(fileSystem, control.getNonArcSourceFileList())) {
      result.add(new SourceFile(BuildType.NON_ARC_BUILD, nonArcSource));
    }
    for (Path supportSource : RelativePaths.fromStrings(fileSystem, control.getSupportFileList())) {
      result.add(new SourceFile(BuildType.NO_BUILD, supportSource));
    }
    if (control.hasInfoplist()) {
      result.add(new SourceFile(
          BuildType.NO_BUILD, RelativePaths.fromString(fileSystem, control.getInfoplist())));
    }
    return result.build();
  }
}
