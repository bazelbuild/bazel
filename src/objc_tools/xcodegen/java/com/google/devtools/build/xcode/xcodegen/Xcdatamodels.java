// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.xcode.util.Equaling;
import com.google.devtools.build.xcode.util.Value;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;

import com.facebook.buck.apple.xcode.xcodeproj.PBXBuildFile;
import com.facebook.buck.apple.xcode.xcodeproj.PBXTarget.ProductType;

import java.nio.file.FileSystem;

/**
 * Contains information about .xcdatamodel directories in an Xcode project.
 */
public class Xcdatamodels extends Value<Xcdatamodels> {

  private final ImmutableSetMultimap<TargetControl, PBXBuildFile> buildFiles;

  private Xcdatamodels(ImmutableSetMultimap<TargetControl, PBXBuildFile> buildFiles) {
    super(buildFiles);
    this.buildFiles = buildFiles;
  }

  /**
   * Map of each build file that should be added to the sources build phase for each target, given
   * the target's control data.
   */
  public ImmutableSetMultimap<TargetControl, PBXBuildFile> buildFiles() {
    return buildFiles;
  }

  public static Xcdatamodels fromTargetControls(
      FileSystem fileSystem, PBXBuildFiles pbxBuildFiles, Iterable<TargetControl> targetControls) {
    ImmutableSetMultimap.Builder<TargetControl, PBXBuildFile> targetLabelToBuildFiles =
        new ImmutableSetMultimap.Builder<>();
    for (TargetControl targetControl : targetControls) {
      Iterable<PBXBuildFile> targetBuildFiles =
          pbxBuildFiles.get(
              AggregateReferenceType.XCVersionGroup,
              RelativePaths.fromStrings(fileSystem, targetControl.getXcdatamodelList()));

      // If this target is not a static library, save the build files. If it's a static lib, we
      // don't need them. The file references we generated with fileObjects will be added to the
      // main group later.
      if (!Equaling.of(
          ProductType.STATIC_LIBRARY, XcodeprojGeneration.productType(targetControl))) {
        targetLabelToBuildFiles.putAll(targetControl, targetBuildFiles);
      }
    }
    return new Xcdatamodels(targetLabelToBuildFiles.build());
  }
}
