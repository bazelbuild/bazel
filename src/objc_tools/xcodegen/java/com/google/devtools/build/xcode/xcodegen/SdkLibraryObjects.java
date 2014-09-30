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
import com.google.common.base.Preconditions;
import com.google.devtools.build.xcode.util.Mapping;
import com.google.devtools.build.xcode.util.Value;

import com.facebook.buck.apple.xcode.xcodeproj.PBXBuildFile;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFrameworksBuildPhase;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A kind of cache which makes it easier to manage PBXFileReference and PBXBuildFile references for
 * frameworks and dylibs that are bundled with the SDK. These are things that are linked with the
 * final binary, some examples being "XCTest.framework", "Foundation.framework", and
 * "libz.dylib".
 */
public final class SdkLibraryObjects implements HasProjectNavigatorFiles {
  private final Map<SdkLibrary, PBXBuildFile> buildFiles;
  private final PBXFileReferences fileReferences;
  private final List<PBXReference> mainGroupReferences;

  /**
   * Represents a single framework or dylib. Contains all information needed to make a corresponding
   * {@link PBXFileReference}.
   */
  public static final class SdkLibrary extends Value<SdkLibrary> {
    private final String pathFromSdkRoot;
    private final String fileType;

    SdkLibrary(String pathFromSdkRoot, String fileType) {
      super(pathFromSdkRoot);
      this.pathFromSdkRoot = pathFromSdkRoot;
      this.fileType = fileType;
    }

    public String getPathFromSdkRoot() {
      return pathFromSdkRoot;
    }

    public String getFileType() {
      return fileType;
    }
  }

  public static SdkLibrary dylib(String name) {
    return new SdkLibrary(String.format("usr/lib/%s.dylib", name), "compiled.mach-o.dylib");
  }

  public static SdkLibrary framework(String name) {
    return new SdkLibrary(
        String.format("System/Library/Frameworks/%s.framework", name), "wrapper.framework");
  }

  public SdkLibraryObjects(PBXFileReferences fileReferences) {
    this.buildFiles = new HashMap<>();
    this.fileReferences = Preconditions.checkNotNull(fileReferences);
    this.mainGroupReferences = new ArrayList<>();
  }

  private PBXFileReference fileReference(SdkLibrary library) {
    PBXFileReference result = fileReferences.get(FileReference.of(
        library.getPathFromSdkRoot(),
        SourceTree.SDKROOT));
    result.setExplicitFileType(Optional.of(library.getFileType()));
    return result;
  }

  /**
   * Returns a build file, creating a new one if it doesn't exist, for the given SDK framework. The
   * SDK framework should not include the ".framework" extension.
   */
  public PBXBuildFile buildFile(SdkLibrary library) {
    for (PBXBuildFile existing : Mapping.of(buildFiles, library).asSet()) {
      return existing;
    }
    PBXFileReference fileRef = fileReference(library);
    mainGroupReferences.add(fileRef);
    PBXBuildFile newBuildFile = new PBXBuildFile(fileRef);
    buildFiles.put(library, newBuildFile);
    return newBuildFile;
  }

  /**
   * Returns a new build phase (in other words, not from a cache) containing the given SDK
   * frameworks. The PBXBuildFile objects <em>are</em> taken from and/or put in the cache.
   */
  public PBXFrameworksBuildPhase newBuildPhase(Iterable<SdkLibrary> libraries) {
    PBXFrameworksBuildPhase buildPhase = new PBXFrameworksBuildPhase();
    for (SdkLibrary library : libraries) {
      buildPhase.getFiles().add(buildFile(library));
    }
    return buildPhase;
  }

  @Override
  public Iterable<PBXReference> mainGroupReferences() {
    return mainGroupReferences;
  }
}
