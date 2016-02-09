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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFile;
import com.google.devtools.build.lib.actions.cache.Digest;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Value for TreeArtifacts, which contains a digest and the {@link FileArtifactValue}s
 * of its child {@link ArtifactFile}s.
 */
public class TreeArtifactValue extends ArtifactValue {
  private final byte[] digest;
  private final Map<PathFragment, FileArtifactValue> childData;

  private TreeArtifactValue(byte[] digest, Map<PathFragment, FileArtifactValue> childData) {
    this.digest = digest;
    this.childData = ImmutableMap.copyOf(childData);
  }

  /**
   * Returns a TreeArtifactValue out of the given Artifact-relative path fragments
   * and their corresponding FileArtifactValues.
   */
  @VisibleForTesting
  public static TreeArtifactValue create(Map<PathFragment, FileArtifactValue> childFileValues) {
    Map<String, Metadata> digestBuilder =
        Maps.newHashMapWithExpectedSize(childFileValues.size());
    for (Map.Entry<PathFragment, FileArtifactValue> e : childFileValues.entrySet()) {
      digestBuilder.put(e.getKey().getPathString(), new Metadata(e.getValue().getDigest()));
    }

    return new TreeArtifactValue(
        Digest.fromMetadata(digestBuilder).asMetadata().digest,
        ImmutableMap.copyOf(childFileValues));
  }

  public FileArtifactValue getSelfData() {
    return FileArtifactValue.createProxy(digest);
  }

  /** Returns the inputs that this artifact expands to, in no particular order. */
  Iterable<ArtifactFile> getChildren(final Artifact base) {
    return ActionInputHelper.asArtifactFiles(base, childData.keySet());
  }

  public Metadata getMetadata() {
    return new Metadata(digest.clone());
  }

  public Set<PathFragment> getChildPaths() {
    return childData.keySet();
  }

  @Nullable
  public byte[] getDigest() {
    return digest.clone();
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
    if (that.digest != digest) {
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
   * A TreeArtifactValue that represents a missing TreeArtifact.
   * This is occasionally useful because Java's concurrent collections disallow null members.
   */
  static final TreeArtifactValue MISSING_TREE_ARTIFACT = new TreeArtifactValue(null,
      ImmutableMap.<PathFragment, FileArtifactValue>of()) {
    @Override
    public FileArtifactValue getSelfData() {
      throw new UnsupportedOperationException();
    }

    @Override
    Iterable<ArtifactFile> getChildren(Artifact base) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Metadata getMetadata() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Set<PathFragment> getChildPaths() {
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
}
