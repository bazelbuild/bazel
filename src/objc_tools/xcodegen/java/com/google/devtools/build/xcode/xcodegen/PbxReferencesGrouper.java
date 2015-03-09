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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.xcode.util.Containing;
import com.google.devtools.build.xcode.util.Equaling;
import com.google.devtools.build.xcode.util.Mapping;

import com.facebook.buck.apple.xcode.xcodeproj.PBXGroup;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;
import com.facebook.buck.apple.xcode.xcodeproj.PBXVariantGroup;

import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * A {@link PBXReference} processor to group self-contained PBXReferences into PBXGroups. Grouping
 * is done to make it easier to navigate the files of the project in Xcode's Project Navigator.
 *
 * <p>A <em>self-contained</em> reference is one that is not a member of a PBXVariantGroup or other
 * aggregate group, although a self-contained reference may contain such a reference as a child.
 *
 * <p>This implementation arranges the {@code PBXFileReference}s into a hierarchy of
 * {@code PBXGroup}s that mirrors the actual location of the files on disk.
 *
 * <p>When using this grouper, the top-level items are the following:
 * <ul>
 *   <li>BUILT_PRODUCTS_DIR - a group containing items in the SourceRoot of this name
 *   <li>SDKROOT - a group containing items that are part of the Xcode install, such as SDK
 *       frameworks
 *   <li>workspace_root - a group containing items within the root of the workspace of the client
 *   <li>miscellaneous - anything that does not belong in one of the above groups is placed directly
 *       in the main group.
 * </ul>
 */
public class PbxReferencesGrouper implements PbxReferencesProcessor {
  private final FileSystem fileSystem;

  public PbxReferencesGrouper(FileSystem fileSystem) {
    this.fileSystem = Preconditions.checkNotNull(fileSystem, "fileSystem");
  }

  /**
   * Converts a {@code String} to a {@code Path} using this instance's file system.
   */
  private Path path(String pathString) {
    return RelativePaths.fromString(fileSystem, pathString);
  }

  /**
   * Returns the deepest directory that contains both paths.
   */
  private Path deepestCommonContainer(Path path1, Path path2) {
    Path container = path("");
    int nameIndex = 0;
    while ((nameIndex < Math.min(path1.getNameCount(), path2.getNameCount()))
        && Equaling.of(path1.getName(nameIndex), path2.getName(nameIndex))) {
      container = container.resolve(path1.getName(nameIndex));
      nameIndex++;
    }
    return container;
  }

  /**
   * Returns the parent of the given path. This is similar to {@link Path#getParent()}, but is
   * nullable-phobic. {@link Path#getParent()} considers the root of the filesystem to be the null
   * Path. This method uses {@code path("")} for the root. This is also how the implementation of
   * {@link PbxReferencesGrouper} expresses <em>root</em> in general.
   */
  private Path parent(Path path) {
    return (path.getNameCount() == 1) ? path("") : path.getParent();
  }

  /**
   * The directory of the PBXGroup that will contain the given reference. For most references, this
   * is just the actual parent directory. For {@code PBXVariantGroup}s, whose children are not
   * guaranteed to be in any common directory except the client root, this returns the deepest
   * common container of each child in the group.
   */
  private Path dirOfContainingPbxGroup(PBXReference reference) {
    if (reference instanceof PBXVariantGroup) {
      PBXVariantGroup variantGroup = (PBXVariantGroup) reference;
      Path path = Paths.get(variantGroup.getChildren().get(0).getPath());
      for (PBXReference child : variantGroup.getChildren()) {
        path = deepestCommonContainer(path, path(child.getPath()));
      }
      return path;
    } else {
      return parent(path(reference.getPath()));
    }
  }

  /**
   * Contains the populated PBXGroups for a certain source tree.
   */
  private class Groups {
    /**
     * Map of paths to the PBXGroup that is used to contain all files and groups in that path.
     */
    final Map<Path, PBXGroup> groupCache;

    Groups(String rootGroupName, SourceTree sourceTree) {
      groupCache = new HashMap<>();
      groupCache.put(path(""), new PBXGroup(rootGroupName, "" /* path */, sourceTree));
    }

    PBXGroup rootGroup() {
      return Mapping.of(groupCache, path("")).get();
    }

    void add(Path dirOfContainingPbxGroup, PBXReference reference) {
      for (PBXGroup container : Mapping.of(groupCache, dirOfContainingPbxGroup).asSet()) {
        container.getChildren().add(reference);
        return;
      }
      PBXGroup newGroup = new PBXGroup(dirOfContainingPbxGroup.getFileName().toString(),
          null /* path */, SourceTree.GROUP);
      newGroup.getChildren().add(reference);
      add(parent(dirOfContainingPbxGroup), newGroup);
      groupCache.put(dirOfContainingPbxGroup, newGroup);
    }
  }

  @Override
  public Iterable<PBXReference> process(Iterable<PBXReference> references) {
    Map<SourceTree, Groups> groupsBySourceTree = ImmutableMap.of(
        SourceTree.GROUP, new Groups("workspace_root", SourceTree.SOURCE_ROOT),
        SourceTree.SDKROOT, new Groups("SDKROOT", SourceTree.SDKROOT),
        SourceTree.BUILT_PRODUCTS_DIR,
            new Groups("BUILT_PRODUCTS_DIR", SourceTree.BUILT_PRODUCTS_DIR));
    ImmutableList.Builder<PBXReference> result = new ImmutableList.Builder<>();

    for (PBXReference reference : references) {
      if (Containing.key(groupsBySourceTree, reference.getSourceTree())) {
        Path containingDir = dirOfContainingPbxGroup(reference);
        Mapping.of(groupsBySourceTree, reference.getSourceTree())
            .get()
            .add(containingDir, reference);
      } else {
        // The reference is not inside any expected source tree, so don't try anything clever. Just
        // add it to the main group directly (not in a nested PBXGroup).
        result.add(reference);
      }
    }

    for (Groups groupsRoot : groupsBySourceTree.values()) {
      if (!groupsRoot.rootGroup().getChildren().isEmpty()) {
        result.add(groupsRoot.rootGroup());
      }
    }

    return result.build();
  }
}
