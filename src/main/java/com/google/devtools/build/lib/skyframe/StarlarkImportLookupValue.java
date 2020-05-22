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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A value that represents a Starlark import lookup result. The lookup value corresponds to exactly
 * one Starlark file, identified by an absolute {@link Label} {@link SkyKey} argument. The Label
 * should not reference the special {@code external} package.
 */
public class StarlarkImportLookupValue implements SkyValue {

  private final Module module; // .bzl module
  private final byte[] transitiveDigest; // of .bzl file and load dependencies

  /**
   * The immediate Starlark file dependency descriptor class corresponding to this value. Using this
   * reference it's possible to reach the transitive closure of Starlark files on which this
   * Starlark file depends.
   */
  private final StarlarkFileDependency dependency;

  @VisibleForTesting
  public StarlarkImportLookupValue(
      Module module, byte[] transitiveDigest, StarlarkFileDependency dependency) {
    this.module = Preconditions.checkNotNull(module);
    this.transitiveDigest = Preconditions.checkNotNull(transitiveDigest);
    this.dependency = Preconditions.checkNotNull(dependency);
  }

  /** Returns the .bzl module. */
  public Module getModule() {
    return module;
  }

  /** Returns the digest of the .bzl module and its transitive load dependencies. */
  public byte[] getTransitiveDigest() {
    return transitiveDigest;
  }

  /** Returns the immediate Starlark file dependency corresponding to this import lookup value. */
  public StarlarkFileDependency getDependency() {
    return dependency;
  }

  private static final Interner<Key> keyInterner = BlazeInterners.newWeakInterner();

  /** SkyKey for a Starlark import. */
  abstract static class Key implements SkyKey {

    /** Returns the label of the .bzl file to be loaded. */
    abstract Label getLabel();

    /**
     * Constructs a new key suitable for evaluating a {@code load()} dependency of this key's .bzl
     * file.
     *
     * <p>The new key uses the given label but the same contextual information -- whether the
     * top-level requesting value is a BUILD or WORKSPACE file, and if it's a WORKSPACE, its
     * chunking info.
     */
    abstract Key getKeyForLoad(Label loadLabel);

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.STARLARK_IMPORTS_LOOKUP;
    }
  }

  /** A key for loading a .bzl during package loading (BUILD evaluation). */
  @Immutable
  @AutoCodec.VisibleForSerialization
  static final class PackageBzlKey extends Key {

    private final Label label;

    private PackageBzlKey(Label label) {
      this.label = Preconditions.checkNotNull(label);
    }

    @Override
    Label getLabel() {
      return label;
    }

    @Override
    Key getKeyForLoad(Label loadLabel) {
      return packageBzlKey(loadLabel);
    }

    @Override
    public String toString() {
      return label.toString();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof PackageBzlKey)) {
        return false;
      }
      return this.label.equals(((PackageBzlKey) obj).label);
    }

    @Override
    public int hashCode() {
      return Objects.hash(PackageBzlKey.class, label);
    }
  }

  /**
   * A key for loading a .bzl during WORKSPACE evaluation.
   *
   * <p>This needs to track "chunking" information, i.e. a sequence number indicating which segment
   * of the WORKSPACE file we are in the process of evaluating. This helps determine the appropriate
   * repository remapping value to use.
   */
  // TODO(brandjon): Question: It looks like the chunk number doesn't play any role in deciding
  // whether or not a repo is available for load()ing. Are we tracking incremental dependencies
  // correctly? For instance, if a repository declaration moves from one workspace chunk to another,
  // are we reevaluating whether its loads are still valid? AI: fix if broken, improve this comment
  // if not broken.
  @Immutable
  @AutoCodec.VisibleForSerialization
  static final class WorkspaceBzlKey extends Key {

    private final Label label;
    private final int workspaceChunk;
    private final RootedPath workspacePath;

    private WorkspaceBzlKey(Label label, int workspaceChunk, RootedPath workspacePath) {
      this.label = Preconditions.checkNotNull(label);
      this.workspaceChunk = workspaceChunk;
      this.workspacePath = Preconditions.checkNotNull(workspacePath);
    }

    @Override
    Label getLabel() {
      return label;
    }

    int getWorkspaceChunk() {
      return workspaceChunk;
    }

    RootedPath getWorkspacePath() {
      return workspacePath;
    }

    @Override
    Key getKeyForLoad(Label loadLabel) {
      return workspaceBzlKey(loadLabel, workspaceChunk, workspacePath);
    }

    @Override
    public String toString() {
      return label + " (in workspace)";
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof WorkspaceBzlKey)) {
        return false;
      }
      WorkspaceBzlKey other = (WorkspaceBzlKey) obj;
      return label.equals(other.label)
          && workspaceChunk == other.workspaceChunk
          && workspacePath.equals(other.workspacePath);
    }

    @Override
    public int hashCode() {
      return Objects.hash(WorkspaceBzlKey.class, label, workspaceChunk, workspacePath);
    }
  }

  /** Constructs a key for loading a regular (non-workspace) .bzl file, from the .bzl's label. */
  static Key packageBzlKey(Label label) {
    return keyInterner.intern(new PackageBzlKey(label));
  }

  /**
   * Constructs a key for loading a .bzl file from the context of evaluating the WORKSPACE file.
   *
   * @param label the label of the bzl file being loaded
   * @param workspaceChunk the workspace chunk that the load statement originated from. If the bzl
   *     file is loaded more than once, this is the chunk that it was first loaded from
   * @param workspacePath the path of the workspace file for the project
   */
  static Key workspaceBzlKey(Label label, int workspaceChunk, RootedPath workspacePath) {
    return keyInterner.intern(new WorkspaceBzlKey(label, workspaceChunk, workspacePath));
  }
}
