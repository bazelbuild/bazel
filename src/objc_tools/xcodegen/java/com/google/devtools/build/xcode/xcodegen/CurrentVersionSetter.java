// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.devtools.build.xcode.plmerge.PlistMerging;

import com.dd.plist.NSDictionary;
import com.dd.plist.NSString;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;
import com.facebook.buck.apple.xcode.xcodeproj.XCVersionGroup;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Processes the {@link XCVersionGroup} instances in a sequence of {@link PBXReference}s to have
 * the {@code currentVersion} field set properly according to the {@code .xccurrentversion} file, if
 * available.
 * 
 * <p>This will NOT set the current version for any group where one of the following is true:
 * <ul>
 *   <li>the sourceTree of the group is not GROUP (meaning it is not relative to the workspace root)
 *   <li>the {@code .xccurrentversion} file does not exist or is not accessible
 *   <li>the plist in the file does not contain the correct entry
 *   <li>the {@link XCVersionGroup} does not have a child that matches the version in the
 *       {@code .xccurrentversion} file
 * </ul>
 */
public final class CurrentVersionSetter implements PbxReferencesProcessor {
  private final Path workspaceRoot;

  public CurrentVersionSetter(Path workspaceRoot) {
    this.workspaceRoot = Preconditions.checkNotNull(workspaceRoot);
  }

  @Override
  public Iterable<PBXReference> process(Iterable<PBXReference> references) {
    for (PBXReference reference : references) {
      if ((reference instanceof XCVersionGroup) && (reference.getPath() != null)) {
        trySetCurrentVersion((XCVersionGroup) reference);
      }
    }
    return references;
  }

  private void trySetCurrentVersion(XCVersionGroup group) {
    if (group.getSourceTree() != SourceTree.GROUP) {
      return;
    }

    Path groupPath = workspaceRoot.resolve(group.getPath());
    Path currentVersionPlist = groupPath.resolve(".xccurrentversion");
    if (!Files.isReadable(currentVersionPlist)) {
      return;
    }

    NSDictionary plist;
    try {
      plist = PlistMerging.readPlistFile(currentVersionPlist);
    } catch (IOException e) {
      return;
    }
    NSString currentVersion = (NSString) plist.get("_XCCurrentVersionName");
    if (currentVersion == null) {
      return;
    }

    for (PBXFileReference child : group.getChildren()) {
      child.setExplicitFileType(Optional.of("wrapper.xcdatamodel"));
      if (child.getName().equals(currentVersion.getContent())) {
        group.setCurrentVersion(Optional.of(child));
      }
    }
  }
}
