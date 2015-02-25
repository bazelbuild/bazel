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

package com.google.devtools.build.xcode.xcodegen.testing;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.xcode.xcodegen.FileReference;

import com.facebook.buck.apple.xcode.xcodeproj.PBXBuildFile;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFrameworksBuildPhase;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;

/**
 * Collection of static utility methods for PBX type transformations, either to other
 * representations of the same or our domain objects.
 */
public class PbxTypes {

  private PbxTypes() {}

  /**
   * Returns all file references the {@code phase} depends on in
   * {@link PBXFrameworksBuildPhase#getFiles()}.
   */
  public static ImmutableList<FileReference> fileReferences(PBXFrameworksBuildPhase phase) {
    return fileReferences(pbxFileReferences(phase));
  }

  /**
   * Transforms the given list of PBX references to file references.
   */
  public static ImmutableList<FileReference> fileReferences(
      Iterable<? extends PBXReference> references) {
    ImmutableList.Builder<FileReference> fileReferences = ImmutableList.builder();
    for (PBXReference reference : references) {
      fileReferences.add(fileReference(reference));
    }
    return fileReferences.build();
  }

  /**
   * Extracts the list of PBX references {@code phase} depends on through
   * {@link PBXFrameworksBuildPhase#getFiles()}.
   */
  public static ImmutableList<PBXReference> pbxFileReferences(PBXFrameworksBuildPhase phase) {
    ImmutableList.Builder<PBXReference> phaseFileReferences = ImmutableList.builder();
    for (PBXBuildFile buildFile : phase.getFiles()) {
      phaseFileReferences.add(buildFile.getFileRef());
    }
    return phaseFileReferences.build();
  }

  /**
   * Converts a PBX file reference to its domain equivalent.
   */
  public static FileReference fileReference(PBXReference reference) {
    FileReference fileReference =
        FileReference.of(reference.getName(), reference.getPath(), reference.getSourceTree());
    if (reference instanceof PBXFileReference) {
      Optional<String> explicitFileType = ((PBXFileReference) reference).getExplicitFileType();
      if (explicitFileType.isPresent()) {
        return fileReference.withExplicitFileType(explicitFileType.get());
      }
    }
    return fileReference;
  }

  /**
   * Returns the string representation of all references the {@code phase} depends on in
   * {@link PBXFrameworksBuildPhase#getFiles()}.
   *
   */
  public static ImmutableList<String> referencePaths(PBXFrameworksBuildPhase phase) {
    return paths(pbxFileReferences(phase));
  }

  /**
   * Transforms the given list of references into their string path representations.
   */
  public static ImmutableList<String> paths(Iterable<? extends PBXReference> references) {
    ImmutableList.Builder<String> paths = ImmutableList.builder();
    for (PBXReference reference : references) {
      paths.add(reference.getPath());
    }
    return paths.build();
  }
}
