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
import com.google.devtools.build.xcode.util.Equaling;
import com.google.devtools.build.xcode.util.Value;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.TargetControl;

import com.facebook.buck.apple.xcode.xcodeproj.PBXBuildFile;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;
import com.facebook.buck.apple.xcode.xcodeproj.PBXResourcesBuildPhase;
import com.facebook.buck.apple.xcode.xcodeproj.PBXTarget.ProductType;

import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

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
  public PBXResourcesBuildPhase resourcesBuildPhase(TargetControl targetControl) {
    PBXResourcesBuildPhase resourcesPhase = new PBXResourcesBuildPhase();
    resourcesPhase.getFiles().addAll(buildFiles().get(targetControl));
    return resourcesPhase;
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
      FileSystem fileSystem, PBXBuildFiles pbxBuildFiles, Iterable<TargetControl> targetControls) {
    ImmutableSetMultimap.Builder<TargetControl, PBXBuildFile> buildFiles =
        new ImmutableSetMultimap.Builder<>();

    for (TargetControl targetControl : targetControls) {
      List<PBXBuildFile> targetBuildFiles = new ArrayList<>();

      Iterable<String> simpleImports =
          Iterables.concat(targetControl.getXcassetsDirList(), targetControl.getBundleImportList());
      // Add .bundle, .xcassets directories to the Project Navigator so they are visible from within
      // Xcode.
      // Bundle imports are handled very similarly to asset catalogs, so we just add them with the
      // same logic. Xcode's automatic file type detection logic is smart enough to see it is a
      // bundle and link it properly, and add the {@code lastKnownFileType} property.
      for (String simpleImport : simpleImports) {
        targetBuildFiles.add(
            pbxBuildFiles.getStandalone(FileReference.of(simpleImport, SourceTree.GROUP)));
      }

      Iterables.addAll(
          targetBuildFiles,
          pbxBuildFiles.get(
              AggregateReferenceType.PBXVariantGroup,
              RelativePaths.fromStrings(fileSystem, targetControl.getGeneralResourceFileList())));

      // If this target is a binary, save the build files. Otherwise, we don't need them. The file
      // references we generated with fileObjects will be added to the main group later.
      if (!Equaling.of(
          ProductType.STATIC_LIBRARY, XcodeprojGeneration.productType(targetControl))) {
        buildFiles.putAll(targetControl, targetBuildFiles);
      }
    }

    return new Resources(buildFiles.build());
  }
}
