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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.xcode.util.Value;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;

import com.facebook.buck.apple.xcode.xcodeproj.PBXBuildFile;
import com.facebook.buck.apple.xcode.xcodeproj.PBXResourcesBuildPhase;

import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Contains information about resources in an Xcode project.
 */
public class Resources extends Value<Resources> {

  private final ImmutableSetMultimap<TargetControl, PBXBuildFile> buildFiles;

  private Resources(ImmutableSetMultimap<TargetControl, PBXBuildFile> buildFiles) {
    super(buildFiles);
    this.buildFiles = buildFiles;
  }

  /**
   * Build files that should be added to the PBXResourcesBuildPhase for the given target.
   */
  public ImmutableSetMultimap<TargetControl, PBXBuildFile> buildFiles() {
    return buildFiles;
  }

  /**
   * Returns the PBXResourcesBuildPhase for the given target, if applicable. It will return an
   * absent {@code Optional} if the target is a library or there are no resources to compile.
   */
  public Optional<PBXResourcesBuildPhase> resourcesBuildPhase(TargetControl targetControl) {
    Set<PBXBuildFile> buildFiles = buildFiles().get(targetControl);
    if (buildFiles.isEmpty()) {
      return Optional.absent();
    }
    PBXResourcesBuildPhase resourcesPhase = new PBXResourcesBuildPhase();
    resourcesPhase.getFiles().addAll(buildFiles);
    return Optional.of(resourcesPhase);
  }

  public static Optional<String> languageOfLprojDir(Path child) {
    Path parent = child.getParent();
    if (parent == null) {
      return Optional.absent();
    }
    String dirName = parent.getFileName().toString();
    String lprojSuffix = ".lproj";
    if (dirName.endsWith(lprojSuffix)) {
      return Optional.of(dirName.substring(0, dirName.length() - lprojSuffix.length()));
    } else {
      return Optional.absent();
    }
  }

  public static Resources fromTargetControls(
      FileSystem fileSystem, FileObjects fileObjects, Iterable<TargetControl> targetControls) {
    ImmutableSetMultimap.Builder<TargetControl, PBXBuildFile> buildFiles =
        new ImmutableSetMultimap.Builder<>();

    for (TargetControl targetControl : targetControls) {
      List<PBXBuildFile> targetBuildFiles = new ArrayList<>();

      // Add .xcassets to the Project Navigator so they can be edited from within Xcode.
      for (String xcassetsDir : targetControl.getXcassetsDirList()) {
        targetBuildFiles.add(
            fileObjects.buildFile(RelativePaths.fromString(fileSystem, xcassetsDir)));
      }

      Iterables.addAll(
          targetBuildFiles,
          fileObjects.buildFilesForAggregates(
              AggregateReferenceType.PBXVariantGroup,
              RelativePaths.fromStrings(fileSystem, targetControl.getGeneralResourceFileList())));

      // If this target is an app, save the build files. Otherwise, we don't need them. The file
      // references we generated with fileObjects will be added to the main group later.
      if (XcodeprojGeneration.isApp(targetControl)) {
        buildFiles.putAll(targetControl, targetBuildFiles);
      }
    }

    return new Resources(buildFiles.build());
  }
}
