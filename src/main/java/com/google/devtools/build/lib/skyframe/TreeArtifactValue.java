// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.HasDigest;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Dirent.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Value for TreeArtifacts, which contains a digest and the {@link FileArtifactValue}s of its child
 * {@link TreeFileArtifact}s.
 */
@AutoCodec
public class TreeArtifactValue implements HasDigest, SkyValue {

  private static final TreeArtifactValue EMPTY =
      new TreeArtifactValue(
          DigestUtils.fromMetadata(ImmutableMap.of()),
          ImmutableSortedMap.of(),
          /* remote= */ false);

  private final byte[] digest;
  private final ImmutableSortedMap<TreeFileArtifact, FileArtifactValue> childData;
  private final boolean remote;

  @AutoCodec.VisibleForSerialization
  TreeArtifactValue(
      byte[] digest,
      ImmutableSortedMap<TreeFileArtifact, FileArtifactValue> childData,
      boolean remote) {
    this.digest = digest;
    this.childData = childData;
    this.remote = remote;
  }

  /**
   * Returns a TreeArtifactValue out of the given Artifact-relative path fragments and their
   * corresponding FileArtifactValues.
   */
  static TreeArtifactValue create(
      Map<TreeFileArtifact, ? extends FileArtifactValue> childFileValues) {
    if (childFileValues.isEmpty()) {
      return EMPTY;
    }
    Map<String, FileArtifactValue> digestBuilder =
        Maps.newHashMapWithExpectedSize(childFileValues.size());
    boolean remote = true;
    for (Map.Entry<TreeFileArtifact, ? extends FileArtifactValue> e : childFileValues.entrySet()) {
      TreeFileArtifact child = e.getKey();
      FileArtifactValue value = e.getValue();
      Preconditions.checkState(
          !FileArtifactValue.OMITTED_FILE_MARKER.equals(value),
          "Cannot construct TreeArtifactValue because child %s was omitted",
          child);
      // TODO(buchgr): Enforce that all children in a tree artifact are either remote or local
      // once b/70354083 is fixed.
      remote = remote && value.isRemote();
      digestBuilder.put(child.getParentRelativePath().getPathString(), value);
    }
    return new TreeArtifactValue(
        DigestUtils.fromMetadata(digestBuilder),
        ImmutableSortedMap.copyOf(childFileValues),
        remote);
  }

  FileArtifactValue getSelfData() {
    return FileArtifactValue.createProxy(digest);
  }

  FileArtifactValue getMetadata() {
    return getSelfData();
  }

  ImmutableSet<PathFragment> getChildPaths() {
    return childData.keySet().stream()
        .map(TreeFileArtifact::getParentRelativePath)
        .collect(toImmutableSet());
  }

  @Nullable
  @Override
  public byte[] getDigest() {
    return digest.clone();
  }

  public ImmutableSet<TreeFileArtifact> getChildren() {
    return childData.keySet();
  }

  ImmutableMap<TreeFileArtifact, FileArtifactValue> getChildValues() {
    return childData;
  }

  /** Returns true if the {@link TreeFileArtifact}s are only stored remotely. */
  public boolean isRemote() {
    return remote;
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(digest);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }

    if (!(other instanceof TreeArtifactValue)) {
      return false;
    }

    TreeArtifactValue that = (TreeArtifactValue) other;
    if (!Arrays.equals(digest, that.digest)) {
      return false;
    }

    return childData.equals(that.childData);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(TreeArtifactValue.class)
        .add("digest", digest)
        .add("childData", childData)
        .toString();
  }

  /**
   * A TreeArtifactValue that represents a missing TreeArtifact. This is occasionally useful because
   * Java's concurrent collections disallow null members.
   */
  static final TreeArtifactValue MISSING_TREE_ARTIFACT =
      new TreeArtifactValue(null, ImmutableSortedMap.of(), /* remote= */ false) {
        @Override
        FileArtifactValue getSelfData() {
          throw new UnsupportedOperationException();
        }

        @Override
        public ImmutableSet<TreeFileArtifact> getChildren() {
          throw new UnsupportedOperationException();
        }

        @Override
        ImmutableMap<TreeFileArtifact, FileArtifactValue> getChildValues() {
          throw new UnsupportedOperationException();
        }

        @Override
        FileArtifactValue getMetadata() {
          throw new UnsupportedOperationException();
        }

        @Override
        ImmutableSet<PathFragment> getChildPaths() {
          throw new UnsupportedOperationException();
        }

        @Nullable
        @Override
        public byte[] getDigest() {
          throw new UnsupportedOperationException();
        }

        @Override
        public int hashCode() {
          return 24; // my favorite number
        }

        @Override
        public boolean equals(Object other) {
          return this == other;
        }

        @Override
        public String toString() {
          return "MISSING_TREE_ARTIFACT";
        }
      };

  private static void explodeDirectory(
      Path treeArtifactPath,
      PathFragment pathToExplode,
      ImmutableSet.Builder<PathFragment> valuesBuilder)
      throws IOException {
    Path dir = treeArtifactPath.getRelative(pathToExplode);
    Collection<Dirent> dirents = dir.readdir(Symlinks.NOFOLLOW);
    for (Dirent dirent : dirents) {
      PathFragment canonicalSubpathFragment = pathToExplode.getChild(dirent.getName());
      if (dirent.getType() == Type.DIRECTORY) {
        explodeDirectory(treeArtifactPath, canonicalSubpathFragment, valuesBuilder);
      } else if (dirent.getType() == Type.SYMLINK) {
        Path subpath = dir.getRelative(dirent.getName());
        PathFragment linkTarget = subpath.readSymbolicLinkUnchecked();
        valuesBuilder.add(canonicalSubpathFragment);
        if (linkTarget.isAbsolute()) {
          // We tolerate absolute symlinks here. They will probably be dangling if any downstream
          // consumer tries to read them, but let that be downstream's problem.
          continue;
        }
        // We visit each path segment of the link target to catch any path traversal outside of the
        // TreeArtifact root directory. For example, for TreeArtifact a/b/c, it is possible to have
        // a symlink, a/b/c/sym_link that points to ../outside_dir/../c/link_target. Although this
        // symlink points to a file under the TreeArtifact, the link target traverses outside of the
        // TreeArtifact into a/b/outside_dir.
        PathFragment intermediatePath = canonicalSubpathFragment.getParentDirectory();
        for (String pathSegment : linkTarget.getSegments()) {
          intermediatePath = intermediatePath.getRelative(pathSegment);
          if (intermediatePath.containsUplevelReferences()) {
            String errorMessage =
                String.format(
                    "A TreeArtifact may not contain relative symlinks whose target paths traverse "
                        + "outside of the TreeArtifact, found %s pointing to %s.",
                    subpath, linkTarget);
            throw new IOException(errorMessage);
          }
        }
      } else if (dirent.getType() == Type.FILE) {
        valuesBuilder.add(canonicalSubpathFragment);
      } else {
        // We shouldn't ever reach here.
        Path subpath = dir.getRelative(dirent.getName());
        throw new IllegalStateException("Could not determine type of file " + subpath);
      }
    }
  }

  /**
   * Recursively get all child files in a directory (excluding child directories themselves, but
   * including all files in them).
   *
   * @throws IOException if there is any problem reading or validating outputs under the given tree
   *     artifact.
   */
  public static Set<PathFragment> explodeDirectory(Path treeArtifactPath) throws IOException {
    ImmutableSet.Builder<PathFragment> explodedDirectory = ImmutableSet.builder();
    explodeDirectory(treeArtifactPath, PathFragment.EMPTY_FRAGMENT, explodedDirectory);
    return explodedDirectory.build();
  }
}
