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

import com.facebook.buck.apple.xcode.xcodeproj.PBXBuildFile;
import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;
import com.facebook.buck.apple.xcode.xcodeproj.PBXVariantGroup;
import com.facebook.buck.apple.xcode.xcodeproj.XCVersionGroup;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.xcode.util.Mapping;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A kind of cache which makes it easier to collect and manage PBXBuildFile and PBXReference
 * objects. It knows how to create new PBXBuildFile, PBXVariantGroup, and PBXFileReference objects
 * from {@link Path} objects and sequences thereof.
 * <p>
 * A PBXFileReference specifies a path to a file and its <em>name</em>. The name is confusingly
 * defined as the real file name for non-localized files (e.g. "foo" in "bar/foo"), and the language
 * name for localized files (e.g. "en" in "bar/en.lproj/foo").
 * <p>
 * A PBXVariantGroup is a set of PBXFileReferences with the same file name (the virtual name). Each
 * file is in some directory named *.lproj. For instance, the following files would belong to the
 * same PBXVariantGroup:
 *
 * <ul>
 *   <li>foo1/en.lproj/file.strings
 *   <li>foo2/ru.lproj/file.strings
 * </ul>
 *
 * Where the virtual name is "file.strings". Note that because of the way PBXVariantGroups are named
 * and specified in .xcodeproj files, it is possible Xcode does or will use it for groups not
 * defined by localization, but we currently ignore that possibility.
 * <p>
 * A PBXBuildFile is the simplest object - it is simply a reference to a PBXReference, which can be
 * either a PBXFileReference or PBXVariantGroup. The fact that PBXFileReference and PBXVariantGroup
 * are considered kinds of PBXReference is reflected in the Java inheritance hierarchy for the
 * classes that model these Xcode objects.
 * <p>
 * The PBXBuildFile is the object referred to in the build phases of targets, so it is seen as a
 * buildable or compilable unit. The PBXFileReference and PBXVariantGroup objects are referred to
 * by the PBXGroups, which define what is visible in the Project Navigator view in Xcode.
 * <p>
 * This class assumes that all paths given through the public API are specified relative to the
 * Xcodegen root, and creates PBXFileReferences that should be added to a group whose path is the
 * root.
 * TODO(bazel-team): Make this an immutable type, of which multiple instances are created as new
 * references and build files are added. The current API is side-effect-based and confusing.
 */
final class PBXBuildFiles implements HasProjectNavigatorFiles {
  /**
   * Map from Paths to the PBXBuildFile that encompasses all and only those Paths. Because the
   * {@link PBXBuildFile}s in this map encompass multiple files, their
   * {@link PBXBuildFile#getFileRef()} value is a {@link PBXVariantGroup} or {@link XCVersionGroup}.
   * <p>
   * Note that this Map reflects the intention of the API, namely that {"a"} does not map to the
   * same thing as {"a", "b"}, and you cannot get a build file with only one of the corresponding
   * files - you need the whole set.
   */
  private Map<ImmutableSet<Path>, PBXBuildFile> aggregateBuildFiles;

  private Map<FileReference, PBXBuildFile> standaloneBuildFiles;
  private PBXFileReferences pbxReferences;
  private List<PBXReference> mainGroupReferences;

  public PBXBuildFiles(PBXFileReferences pbxFileReferences) {
    this.aggregateBuildFiles = new HashMap<>();
    this.standaloneBuildFiles = new HashMap<>();
    this.pbxReferences = Preconditions.checkNotNull(pbxFileReferences);
    this.mainGroupReferences = new ArrayList<>();
  }

  private PBXBuildFile aggregateBuildFile(ImmutableSet<Path> paths, PBXReference reference) {
    Preconditions.checkArgument(!paths.isEmpty(), "paths must be non-empty");
    for (PBXBuildFile cached : Mapping.of(aggregateBuildFiles, paths).asSet()) {
      return cached;
    }
    PBXBuildFile buildFile = new PBXBuildFile(reference);
    mainGroupReferences.add(reference);
    aggregateBuildFiles.put(paths, buildFile);
    return buildFile;
  }

  /**
   * Returns new or cached instances of PBXBuildFiles corresponding to files that may or may not
   * belong to an aggregate reference (see {@link AggregateReferenceType}). Files specified by the
   * {@code paths} argument are grouped into individual PBXBuildFiles using the given {@link
   * AggregateReferenceType}. Files that are standalone are not put in an aggregate reference, but
   * are put in a standalone PBXBuildFile in the returned sequence.
   */
  public Iterable<PBXBuildFile> get(AggregateReferenceType type, Collection<Path> paths) {
    ImmutableList.Builder<PBXBuildFile> result = new ImmutableList.Builder<>();
    SetMultimap<AggregateKey, Path> keyedPaths = type.aggregates(paths);
    for (Map.Entry<AggregateKey, Collection<Path>> aggregation : keyedPaths.asMap().entrySet()) {
      if (!aggregation.getKey().isStandalone()) {
        ImmutableSet<Path> itemPaths = ImmutableSet.copyOf(aggregation.getValue());
        result.add(aggregateBuildFile(
            itemPaths, type.create(aggregation.getKey(), fileReferences(itemPaths))));
      }
    }
    for (Path generalResource : keyedPaths.get(AggregateKey.standalone())) {
      result.add(getStandalone(FileReference.of(generalResource.toString(), SourceTree.GROUP)));
    }

    return result.build();
  }

  /**
   * Returns a new or cached instance of a PBXBuildFile for a file that is not part of a variant
   * group.
   */
  public PBXBuildFile getStandalone(FileReference file) {
    for (PBXBuildFile cached : Mapping.of(standaloneBuildFiles, file).asSet()) {
      return cached;
    }
    PBXBuildFile buildFile = new PBXBuildFile(pbxReferences.get(file));
    mainGroupReferences.add(pbxReferences.get(file));
    standaloneBuildFiles.put(file, buildFile);
    return buildFile;
  }

  /** Applies {@link #fileReference(Path)} to each item in the sequence. */
  private final Iterable<PBXFileReference> fileReferences(Collection<Path> paths) {
    ImmutableList.Builder<PBXFileReference> result = new ImmutableList.Builder<>();
    for (Path path : paths) {
      result.add(fileReference(path));
    }
    return result.build();
  }

  /**
   * Returns a new or cached PBXFileReference for the given file. The name of the reference depends
   * on whether the file is in a localized (*.lproj) directory. If it is localized, then the name
   * of the reference is the name of the language (the text before ".lproj"). Otherwise, the name is
   * the same as the file name (e.g. Localizable.strings). This is confusing, but it is how Xcode
   * creates PBXFileReferences.
   */
  private PBXFileReference fileReference(Path path) {
    Optional<String> language = Resources.languageOfLprojDir(path);
    String name = language.isPresent() ? language.get() : path.getFileName().toString();
    return pbxReferences.get(FileReference.of(name, path.toString(), SourceTree.GROUP));
  }

  @Override
  public Iterable<PBXReference> mainGroupReferences() {
    return mainGroupReferences;
  }
}
