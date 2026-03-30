// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/** A collection of {@link FilesetOutputSymlink}s comprising the output tree of a fileset. */
public final class FilesetOutputTree implements RichArtifactData {

  public static final FilesetOutputTree EMPTY =
      new FilesetOutputTree(ImmutableList.of(), ImmutableMap.of(), false);

  public static FilesetOutputTree forward(FilesetOutputTree other) {
    return other.isEmpty()
        ? EMPTY
        : new FilesetOutputTree(other.symlinks, other.treeArtifacts, true);
  }

  public static FilesetOutputTree create(
      List<FilesetOutputSymlink> symlinks, Map<Artifact, TreeArtifactValue> treeArtifacts) {
    ImmutableList<FilesetOutputSymlink> sortedSymlinks =
        ImmutableList.sortedCopyOf(Comparator.comparing(FilesetOutputSymlink::name), symlinks);
    return symlinks.isEmpty()
        ? EMPTY
        : new FilesetOutputTree(sortedSymlinks, ImmutableMap.copyOf(treeArtifacts), false);
  }

  private final ImmutableList<FilesetOutputSymlink> symlinks;
  private final ImmutableMap<Artifact, TreeArtifactValue> treeArtifacts;
  private final boolean forwarded;

  private FilesetOutputTree(
      ImmutableList<FilesetOutputSymlink> symlinks,
      ImmutableMap<Artifact, TreeArtifactValue> treeArtifacts,
      boolean forwarded) {
    this.symlinks = checkNotNull(symlinks);
    this.treeArtifacts = checkNotNull(treeArtifacts);
    this.forwarded = forwarded;
  }

  /** Returns the symlinks in the fileset, ordered by {@link FilesetOutputSymlink#name()}. */
  public ImmutableList<FilesetOutputSymlink> symlinks() {
    return symlinks;
  }

  /**
   * Returns true if this Fileset is really created from a different action.
   *
   * <p>This is used to avoid double-counting the size of the fileset in metrics.
   */
  public boolean isForwarded() {
    return forwarded;
  }

  public int size() {
    return symlinks.size();
  }

  public boolean isEmpty() {
    return symlinks.isEmpty();
  }

  /**
   * Returns the metadata of all tree artifacts included in this fileset.
   *
   * <p>Individual children of these tree artifacts each have their own entry in {@link
   * #symlinks()}, unless they were excluded by the {@code excludes} parameter on {@code
   * FilesetEntry}.
   */
  public ImmutableMap<Artifact, TreeArtifactValue> getTreeArtifacts() {
    return treeArtifacts;
  }

  @Override
  public int hashCode() {
    return symlinks.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof FilesetOutputTree that)) {
      return false;
    }
    return symlinks.equals(that.symlinks);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("symlinks", symlinks).toString();
  }

  public void addTo(Fingerprint fp) {
    for (var symlink : symlinks) {
      fp.addPath(symlink.name());
      fp.addPath(symlink.target().getExecPath());
      fp.addBytes(symlink.metadata().getDigest());
    }
  }
}
