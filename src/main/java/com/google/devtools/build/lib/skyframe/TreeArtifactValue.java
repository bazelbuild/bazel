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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.HasDigest;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Value for TreeArtifacts, which contains a digest and the {@link FileArtifactValue}s of its child
 * {@link TreeFileArtifact}s.
 */
public class TreeArtifactValue implements HasDigest, SkyValue {

  /** Returns an empty {@link TreeArtifactValue}. */
  public static TreeArtifactValue empty() {
    return EMPTY;
  }

  /**
   * Returns a new {@link Builder} for the given parent tree artifact.
   *
   * <p>The returned builder only supports adding children under this parent. To build multiple tree
   * artifacts at once, use {@link MultiBuilder}.
   */
  public static Builder newBuilder(SpecialArtifact parent) {
    return new Builder(parent);
  }

  /** Builder for constructing multiple instances of {@link TreeArtifactValue} at once. */
  public interface MultiBuilder {
    /**
     * Puts a child tree file into this builder under its {@linkplain TreeFileArtifact#getParent
     * parent}.
     *
     * @return {@code this} for convenience
     */
    MultiBuilder putChild(TreeFileArtifact child, FileArtifactValue metadata);

    /**
     * For each unique parent seen by this builder, passes the aggregated metadata to {@link
     * TreeArtifactInjector#injectTree}.
     */
    void injectTo(TreeArtifactInjector treeInjector);
  }

  /** Returns a new {@link MultiBuilder}. */
  public static MultiBuilder newMultiBuilder() {
    return new BasicMultiBuilder();
  }

  /** Returns a new thread-safe {@link MultiBuilder}. */
  public static MultiBuilder newConcurrentMultiBuilder() {
    return new ConcurrentMultiBuilder();
  }

  @SerializationConstant @AutoCodec.VisibleForSerialization
  static final TreeArtifactValue EMPTY =
      new TreeArtifactValue(
          DigestUtils.fromMetadata(ImmutableMap.of()),
          ImmutableSortedMap.of(),
          /*entirelyRemote=*/ false);

  private final byte[] digest;
  private final ImmutableSortedMap<TreeFileArtifact, FileArtifactValue> childData;
  private final boolean entirelyRemote;

  private TreeArtifactValue(
      byte[] digest,
      ImmutableSortedMap<TreeFileArtifact, FileArtifactValue> childData,
      boolean entirelyRemote) {
    this.digest = digest;
    this.childData = childData;
    this.entirelyRemote = entirelyRemote;
  }

  FileArtifactValue getMetadata() {
    return FileArtifactValue.createProxy(digest);
  }

  ImmutableSet<PathFragment> getChildPaths() {
    return childData.keySet().stream()
        .map(TreeFileArtifact::getParentRelativePath)
        .collect(toImmutableSet());
  }

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
  public boolean isEntirelyRemote() {
    return entirelyRemote;
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
    return MoreObjects.toStringHelper(this)
        .add("digest", digest)
        .add("childData", childData)
        .toString();
  }

  /**
   * Represents a tree artifact that was intentionally omitted, similar to {@link
   * FileArtifactValue#OMITTED_FILE_MARKER}.
   */
  @SerializationConstant
  public static final TreeArtifactValue OMITTED_TREE_MARKER = createMarker("OMITTED_TREE_MARKER");

  /**
   * A TreeArtifactValue that represents a missing TreeArtifact. This is occasionally useful because
   * Java's concurrent collections disallow null members.
   */
  static final TreeArtifactValue MISSING_TREE_ARTIFACT = createMarker("MISSING_TREE_ARTIFACT");

  private static TreeArtifactValue createMarker(String toStringRepresentation) {
    return new TreeArtifactValue(null, ImmutableSortedMap.of(), /*entirelyRemote=*/ false) {
      @Override
      public ImmutableSet<TreeFileArtifact> getChildren() {
        throw new UnsupportedOperationException(toString());
      }

      @Override
      ImmutableMap<TreeFileArtifact, FileArtifactValue> getChildValues() {
        throw new UnsupportedOperationException(toString());
      }

      @Override
      FileArtifactValue getMetadata() {
        throw new UnsupportedOperationException(toString());
      }

      @Override
      ImmutableSet<PathFragment> getChildPaths() {
        throw new UnsupportedOperationException(toString());
      }

      @Nullable
      @Override
      public byte[] getDigest() {
        throw new UnsupportedOperationException(toString());
      }

      @Override
      public int hashCode() {
        return System.identityHashCode(this);
      }

      @Override
      public boolean equals(Object other) {
        return this == other;
      }

      @Override
      public String toString() {
        return toStringRepresentation;
      }
    };
  }

  /** Visitor for use in {@link #visitTree}. */
  @FunctionalInterface
  public interface TreeArtifactVisitor {
    /**
     * Called for every directory entry encountered during tree traversal.
     *
     * <p>Symlinks are not followed during traversal and are simply reported as {@link
     * Dirent.Type#SYMLINK} regardless of whether they point to a file, directory, or are dangling.
     *
     * <p>{@code type} is guaranteed never to be {@link Dirent.Type#UNKNOWN}, since if this type is
     * encountered, {@link IOException} is immediately thrown without invoking the visitor.
     *
     * <p>If the implementation throws {@link IOException}, traversal is immediately halted and the
     * exception is propagated.
     */
    void visit(PathFragment parentRelativePath, Dirent.Type type) throws IOException;
  }

  /**
   * Recursively visits all descendants under a directory.
   *
   * <p>{@link TreeArtifactVisitor#visit} is invoked on {@code visitor} for each directory, file,
   * and symlink under the given {@code parentDir}.
   *
   * <p>This method is intended to provide uniform semantics for constructing a tree artifact,
   * including special logic that validates directory entries. Invalid directory entries include a
   * symlink that traverses outside of the tree artifact and any entry of {@link
   * Dirent.Type#UNKNOWN}, such as a named pipe.
   *
   * @throws IOException if there is any problem reading or validating outputs under the given tree
   *     artifact directory, or if {@link TreeArtifactVisitor#visit} throws {@link IOException}
   */
  public static void visitTree(Path parentDir, TreeArtifactVisitor visitor) throws IOException {
    visitTree(parentDir, PathFragment.EMPTY_FRAGMENT, checkNotNull(visitor));
  }

  private static void visitTree(Path parentDir, PathFragment subdir, TreeArtifactVisitor visitor)
      throws IOException {
    for (Dirent dirent : parentDir.getRelative(subdir).readdir(Symlinks.NOFOLLOW)) {
      PathFragment parentRelativePath = subdir.getChild(dirent.getName());
      Dirent.Type type = dirent.getType();

      if (type == Dirent.Type.UNKNOWN) {
        throw new IOException(
            "Could not determine type of file for " + parentRelativePath + " under " + parentDir);
      }

      if (type == Dirent.Type.SYMLINK) {
        checkSymlink(subdir, parentDir.getRelative(parentRelativePath));
      }

      visitor.visit(parentRelativePath, type);

      if (type == Dirent.Type.DIRECTORY) {
        visitTree(parentDir, parentRelativePath, visitor);
      }
    }
  }

  private static void checkSymlink(PathFragment subDir, Path path) throws IOException {
    PathFragment linkTarget = path.readSymbolicLinkUnchecked();
    if (linkTarget.isAbsolute()) {
      // We tolerate absolute symlinks here. They will probably be dangling if any downstream
      // consumer tries to read them, but let that be downstream's problem.
      return;
    }

    // Visit each path segment of the link target to catch any path traversal outside of the
    // TreeArtifact root directory. For example, for TreeArtifact a/b/c, it is possible to have a
    // symlink, a/b/c/sym_link that points to ../outside_dir/../c/link_target. Although this symlink
    // points to a file under the TreeArtifact, the link target traverses outside of the
    // TreeArtifact into a/b/outside_dir.
    PathFragment intermediatePath = subDir;
    for (String pathSegment : linkTarget.getSegments()) {
      intermediatePath = intermediatePath.getRelative(pathSegment);
      if (intermediatePath.containsUplevelReferences()) {
        String errorMessage =
            String.format(
                "A TreeArtifact may not contain relative symlinks whose target paths traverse "
                    + "outside of the TreeArtifact, found %s pointing to %s.",
                path, linkTarget);
        throw new IOException(errorMessage);
      }
    }
  }

  /** Builder for a {@link TreeArtifactValue}. */
  public static final class Builder {
    private final ImmutableSortedMap.Builder<TreeFileArtifact, FileArtifactValue> childData =
        ImmutableSortedMap.naturalOrder();
    private final SpecialArtifact parent;

    Builder(SpecialArtifact parent) {
      checkArgument(parent.isTreeArtifact(), "%s is not a tree artifact", parent);
      this.parent = parent;
    }

    /**
     * Adds a child to this builder.
     *
     * <p>The child's {@linkplain TreeFileArtifact#getParent parent} <em>must</em> match the parent
     * with which this builder was initialized.
     *
     * <p>Children may be added in any order. The children are sorted prior to constructing the
     * final {@link TreeArtifactValue}.
     *
     * <p>It is illegal to call this method with {@link FileArtifactValue.OMITTED_FILE_MARKER}. When
     * children are omitted, use {@link TreeArtifactValue#OMITTED_TREE_MARKER}.
     *
     * @return {@code this} for convenience
     */
    public Builder putChild(TreeFileArtifact child, FileArtifactValue metadata) {
      checkArgument(
          child.getParent().equals(parent),
          "While building TreeArtifactValue for %s, got %s with parent %s",
          parent,
          child,
          child.getParent());
      checkArgument(
          !FileArtifactValue.OMITTED_FILE_MARKER.equals(metadata),
          "Cannot construct TreeArtifactValue for %s because child %s was omitted",
          parent,
          child);
      childData.put(child, metadata);
      return this;
    }

    /** Builds the final {@link TreeArtifactValue}. */
    public TreeArtifactValue build() {
      ImmutableSortedMap<TreeFileArtifact, FileArtifactValue> finalChildData = childData.build();
      if (finalChildData.isEmpty()) {
        return EMPTY;
      }

      Fingerprint fingerprint = new Fingerprint();
      boolean entirelyRemote = true;

      for (Map.Entry<TreeFileArtifact, FileArtifactValue> childData : finalChildData.entrySet()) {
        // Digest will be deterministic because children are sorted.
        fingerprint.addPath(childData.getKey().getParentRelativePath());
        childData.getValue().addTo(fingerprint);

        // Tolerate a mix of local and remote children (b/152496153#comment80).
        entirelyRemote &= childData.getValue().isRemote();
      }

      return new TreeArtifactValue(fingerprint.digestAndReset(), finalChildData, entirelyRemote);
    }
  }

  private static final class BasicMultiBuilder implements MultiBuilder {
    private final Map<SpecialArtifact, Builder> map = new HashMap<>();

    @Override
    public MultiBuilder putChild(TreeFileArtifact child, FileArtifactValue metadata) {
      map.computeIfAbsent(child.getParent(), Builder::new).putChild(child, metadata);
      return this;
    }

    @Override
    public void injectTo(TreeArtifactInjector treeInjector) {
      map.forEach((parent, builder) -> treeInjector.injectTree(parent, builder.build()));
    }
  }

  @ThreadSafe
  private static final class ConcurrentMultiBuilder implements MultiBuilder {
    private final ConcurrentMap<SpecialArtifact, ConcurrentMap<TreeFileArtifact, FileArtifactValue>>
        map = new ConcurrentHashMap<>();

    @Override
    public MultiBuilder putChild(TreeFileArtifact child, FileArtifactValue metadata) {
      map.computeIfAbsent(child.getParent(), parent -> new ConcurrentHashMap<>())
          .put(child, metadata);
      return this;
    }

    @Override
    public void injectTo(TreeArtifactInjector treeInjector) {
      map.forEach(
          (parent, children) -> {
            Builder builder = new Builder(parent);
            children.forEach(builder::putChild);
            treeInjector.injectTree(parent, builder.build());
          });
    }
  }
}
