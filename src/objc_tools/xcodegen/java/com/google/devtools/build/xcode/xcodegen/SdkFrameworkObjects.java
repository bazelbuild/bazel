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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.devtools.build.xcode.util.Mapping;

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
 * frameworks. A framework is something that is linked with the final binary, some examples being
 * "XCTest.framework" and "Foundation.framework".
 */
public final class SdkFrameworkObjects implements HasProjectNavigatorFiles {
  private final Map<String, PBXBuildFile> buildFiles;
  private final PBXFileReferences fileReferences;
  private final List<PBXReference> mainGroupReferences;

  public SdkFrameworkObjects(PBXFileReferences fileReferences) {
    this.buildFiles = new HashMap<>();
    this.fileReferences = Preconditions.checkNotNull(fileReferences);
    this.mainGroupReferences = new ArrayList<>();
  }

  @VisibleForTesting
  static String pathFromSdkRoot(String name) {
    return String.format("System/Library/Frameworks/%s.framework", name);
  }

  private PBXFileReference sdkFramework(String name) {
    PBXFileReference result = fileReferences.get(FileReference.of(
        pathFromSdkRoot(name),
        SourceTree.SDKROOT));
    result.setExplicitFileType(Optional.of("wrapper.framework"));
    return result;
  }

  /**
   * Returns a build file, creating a new one if it doesn't exist, for the given SDK framework. The
   * SDK framework should not include the ".framework" extension.
   */
  public PBXBuildFile buildFile(String sdkFrameworkName) {
    for (PBXBuildFile existing : Mapping.of(buildFiles, sdkFrameworkName).asSet()) {
      return existing;
    }
    PBXFileReference fileRef = sdkFramework(sdkFrameworkName);
    mainGroupReferences.add(fileRef);
    PBXBuildFile newBuildFile = new PBXBuildFile(fileRef);
    buildFiles.put(sdkFrameworkName, newBuildFile);
    return newBuildFile;
  }

  /**
   * Returns a new build phase (in other words, not from a cache) containing the given SDK
   * frameworks. The PBXBuildFile objects <em>are</em> taken from and/or put in the cache.
   */
  public PBXFrameworksBuildPhase newBuildPhase(Iterable<String> sdkFrameworkNames) {
    PBXFrameworksBuildPhase buildPhase = new PBXFrameworksBuildPhase();
    for (String sdkFrameworkName : sdkFrameworkNames) {
      buildPhase.getFiles().add(buildFile(sdkFrameworkName));
    }
    return buildPhase;
  }

  @Override
  public Iterable<PBXReference> mainGroupReferences() {
    return mainGroupReferences;
  }
}
