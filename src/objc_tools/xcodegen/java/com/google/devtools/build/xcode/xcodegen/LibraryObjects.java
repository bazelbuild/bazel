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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;

import com.facebook.buck.apple.xcode.xcodeproj.PBXBuildFile;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFrameworksBuildPhase;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;

/**
 * Collector that gathers references to libraries and frameworks when generating {@linkplain
 * PBXFrameworksBuildPhase framework build phases} for later display in XCode.
 *
 * <p>Use this class to {@link #newBuildPhase() generate} a framework build phase for each target,
 * by adding things that are linked with the final binary:
 * {@link BuildPhaseBuilder#addFramework frameworks} ("v1_6/GoogleMaps.framework"),
 * {@link BuildPhaseBuilder#addSdkFramework SDK frameworks} ("XCTest.framework",
 * "Foundation.framework") or {@link BuildPhaseBuilder#addDylib dylibs} ("libz.dylib"). Anything
 * added here will also be returned by {@link #mainGroupReferences}.
 *
 * <p>File references used by this class are de-duplicated against a global cache.
 */
public final class LibraryObjects implements HasProjectNavigatorFiles {

  @VisibleForTesting static final String FRAMEWORK_FILE_TYPE = "wrapper.framework";
  @VisibleForTesting static final String DYLIB_FILE_TYPE = "compiled.mach-o.dylib";

  private final LinkedHashMap<FileReference, PBXReference> fileToMainGroupReferences =
      new LinkedHashMap<>();
  private final PBXFileReferences fileReferenceCache;

  /**
   * @param fileReferenceCache global file reference repository used to avoid creating the same
   *    file reference twice
   */
  public LibraryObjects(PBXFileReferences fileReferenceCache) {
    this.fileReferenceCache = checkNotNull(fileReferenceCache);
  }

  /**
   * Builder that assembles information required to generate a {@link PBXFrameworksBuildPhase}.
   */
  public final class BuildPhaseBuilder {

    private final LinkedHashSet<FileReference> fileReferences = new LinkedHashSet<>();

    private BuildPhaseBuilder() {} // Don't allow instantiation from outside the enclosing class.

    /**
     * Creates a new dylib library based on the passed name.
     *
     * @param name simple dylib without ".dylib" suffix, e.g. "libz"
     */
    public BuildPhaseBuilder addDylib(String name) {
      FileReference reference =
          FileReference.of(String.format("usr/lib/%s.dylib", name), SourceTree.SDKROOT)
              .withExplicitFileType(DYLIB_FILE_TYPE);
      fileReferences.add(reference);
      return this;
    }

    /**
     * Creates a new SDK framework based on the passed name.
     *
     * @param name simple framework name without ".framework" suffix, e.g. "Foundation"
     */
    public BuildPhaseBuilder addSdkFramework(String name) {
      String location = String.format("System/Library/Frameworks/%s.framework", name);
      FileReference reference =
          FileReference.of(location, SourceTree.SDKROOT).withExplicitFileType(FRAMEWORK_FILE_TYPE);
      fileReferences.add(reference);
      return this;
    }

    /**
     * Creates a new (non-SDK) framework based on the given path.
     *
     * @param execPath path to the framework's folder, relative to the xcodeproject's path root,
     *    e.g. "v1_6/GoogleMaps.framework"
     */
    public BuildPhaseBuilder addFramework(String execPath) {
      FileReference reference =
          FileReference.of(execPath, SourceTree.GROUP).withExplicitFileType(FRAMEWORK_FILE_TYPE);
      fileReferences.add(reference);
      return this;
    }

    /**
     * Returns a new build phase containing the added libraries.
     */
    public PBXFrameworksBuildPhase build() {
      PBXFrameworksBuildPhase buildPhase = new PBXFrameworksBuildPhase();
      for (FileReference reference : fileReferences) {
        PBXFileReference fileRef = fileReferenceCache.get(reference);
        buildPhase.getFiles().add(new PBXBuildFile(fileRef));
        fileToMainGroupReferences.put(reference, fileRef);
      }
      return buildPhase;
    }
  }

  /**
   * Returns a builder for a new build phase.
   */
  public BuildPhaseBuilder newBuildPhase() {
    return new BuildPhaseBuilder();
  }

  @Override
  public Iterable<PBXReference> mainGroupReferences() {
    return fileToMainGroupReferences.values();
  }
}
