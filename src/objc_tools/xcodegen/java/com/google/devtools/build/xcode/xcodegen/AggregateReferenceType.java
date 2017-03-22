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

import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;
import com.facebook.buck.apple.xcode.xcodeproj.PBXVariantGroup;
import com.facebook.buck.apple.xcode.xcodeproj.XCVersionGroup;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.SetMultimap;
import java.nio.file.Path;
import java.util.Collection;

/**
 * An aggregate reference is a kind of PBXReference that contains one or more files, grouped by some
 * criteria, and appearing as a group in the Xcode project navigator, and often handled as a single
 * file during the build phase and in other situations.
 */
public enum AggregateReferenceType {
  /**
   * A group which contains multiple .xcdatamodel directories where each is a different version of
   * the same schema. We may have to support other files besides .xcdatamodel in the future.
   * Instances of this group are represented by {@link XCVersionGroup} and are grouped by the
   * relative path of the containing .xcdatamodeld directory.
   */
  XCVersionGroup {
    @Override
    public PBXReference create(AggregateKey key, Iterable<PBXFileReference> children) {
      XCVersionGroup result = new XCVersionGroup(
          key.name().orNull(),
          key.path().isPresent() ? key.path().get().toString() : null,
          SourceTree.GROUP);
      Iterables.addAll(result.getChildren(), children);
      return result;
    }

    @Override
    public AggregateKey aggregateKey(Path path) {
      Path parent = path.getParent();
      if (parent.getFileName().toString().endsWith(".xcdatamodeld")) {
        return new AggregateKey(
            Optional.of(parent.getFileName().toString()), Optional.of(parent));
      } else {
        return AggregateKey.standalone();
      }
    }

    @Override
    public Path pathInAggregate(Path path) {
      return path.getFileName();
    }
  },

  /**
   * A group which contains the same content in multiple languages, each language belonging to a
   * different file. Instances of this group are represented by {@link PBXVariantGroup} and are
   * grouped by the base name of the file (e.g. "foo" in "/usr/bar/foo").
   */
  PBXVariantGroup {
    @Override
    public PBXReference create(AggregateKey key, Iterable<PBXFileReference> children) {
      PBXVariantGroup result = new PBXVariantGroup(
          key.name().orNull(),
          key.path().isPresent() ? key.path().get().toString() : null,
          SourceTree.GROUP);
      Iterables.addAll(result.getChildren(), children);
      return result;
    }

    @Override
    public AggregateKey aggregateKey(Path path) {
      if (Resources.languageOfLprojDir(path).isPresent()) {
        return new AggregateKey(
            Optional.of(path.getFileName().toString()), Optional.<Path>absent());
      } else {
        return AggregateKey.standalone();
      }
    }

    @Override
    public Path pathInAggregate(Path path) {
      return path;
    }
  };

  /**
   * Creates a new instance of this group with the group information and children.
   */
  public abstract PBXReference create(AggregateKey key, Iterable<PBXFileReference> children);

  /**
   * Returns the value by which this item should be grouped. All items sharing the same key should
   * belong to the same group. An {@link AggregateKey#standalone()} return here indicates that the
   * item should not belong to a group and should be built and treated as a standalone file.
   */
  public abstract AggregateKey aggregateKey(Path path);

  public abstract Path pathInAggregate(Path path);

  /** Groups a sequence of items according to their {@link #aggregateKey(Path)}. */
  public SetMultimap<AggregateKey, Path> aggregates(Collection<Path> paths) {
    ImmutableSetMultimap.Builder<AggregateKey, Path> result =
        new ImmutableSetMultimap.Builder<>();
    for (Path path : paths) {
      AggregateKey key = aggregateKey(path);
      Path referencePath = key.isStandalone() ? path : pathInAggregate(path);
      result.put(key, referencePath);
    }
    return result.build();
  }
}
