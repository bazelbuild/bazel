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
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import net.starlark.java.eval.Module;

/**
 * A value that represents the .bzl module loaded by a Starlark {@code load()} statement.
 *
 * <p>The key consists of an absolute {@link Label} and the context in which the load occurs. The
 * Label should not reference the special {@code external} package.
 *
 * <p>This value is also used to represent the special prelude file that may be implicitly loaded
 * and sourced by BUILD files. The prelude file need not end in ".bzl".
 */
public class BzlLoadValue implements SkyValue {

  private final Module module; // .bzl module (and indirectly, the entire load DAG)
  private final byte[] transitiveDigest; // of .bzl file and load dependencies

  @VisibleForTesting
  public BzlLoadValue(Module module, byte[] transitiveDigest) {
    this.module = Preconditions.checkNotNull(module);
    this.transitiveDigest = Preconditions.checkNotNull(transitiveDigest);
  }

  /** Returns the .bzl module. */
  public Module getModule() {
    return module;
  }

  /** Returns the digest of the .bzl module and its transitive load dependencies. */
  public byte[] getTransitiveDigest() {
    return transitiveDigest;
  }

  private static final Interner<Key> keyInterner = BlazeInterners.newWeakInterner();

  /** SkyKey for a Starlark load. */
  abstract static class Key implements SkyKey {

    /**
     * Returns the label of the .bzl file to be loaded.
     *
     * <p>For {@link KeyForBuiltins}, it must begin with {@code @_builtins//:}. (It is legal for
     * other keys to use {@code @_builtins}, but since no real repo by that name may be defined,
     * they won't evaluate to a successful result.)
     */
    abstract Label getLabel();

    /** Returns true if this is a request for the special BUILD prelude file. */
    boolean isBuildPrelude() {
      return false;
    }

    /**
     * Constructs a new key suitable for evaluating a {@code load()} dependency of this key's .bzl
     * file.
     *
     * <p>The new key uses the given label but the same contextual information -- whether the
     * top-level requesting value is a BUILD or WORKSPACE file, and if it's a WORKSPACE, its
     * chunking info.
     */
    abstract Key getKeyForLoad(Label loadLabel);

    /**
     * Constructs an BzlCompileValue key suitable for retrieving the Starlark code for this .bzl,
     * given the Root in which to find its file.
     */
    abstract BzlCompileValue.Key getCompileKey(Root root);

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.BZL_LOAD;
    }
  }

  /** A key for loading a .bzl during package loading (BUILD evaluation). */
  @Immutable
  @AutoCodec.VisibleForSerialization
  static final class KeyForBuild extends Key {

    private final Label label;

    /**
     * True if this is the special prelude file, whose declarations are implicitly loaded by all
     * BUILD files.
     */
    private final boolean isBuildPrelude;

    private KeyForBuild(Label label, boolean isBuildPrelude) {
      this.label = Preconditions.checkNotNull(label);
      this.isBuildPrelude = isBuildPrelude;
    }

    @Override
    Label getLabel() {
      return label;
    }

    @Override
    boolean isBuildPrelude() {
      return isBuildPrelude;
    }

    @Override
    Key getKeyForLoad(Label loadLabel) {
      // Note that the returned key always has !isBuildPrelude. I.e., if the prelude file loads
      // another .bzl, the loaded .bzl is processed as normal with no special prelude magic. This is
      // because 1) only the prelude file, not its dependencies, should automatically re-export its
      // loaded symbols; and 2) we don't want prelude-loaded modules to end up cloned if they're
      // also loaded through normal means.
      return keyForBuild(loadLabel);
    }

    @Override
    BzlCompileValue.Key getCompileKey(Root root) {
      if (isBuildPrelude) {
        return BzlCompileValue.keyForBuildPrelude(root, label);
      } else {
        return BzlCompileValue.key(root, label);
      }
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
      if (!(obj instanceof KeyForBuild)) {
        return false;
      }
      KeyForBuild other = (KeyForBuild) obj;
      return this.label.equals(other.label) && this.isBuildPrelude == other.isBuildPrelude;
    }

    @Override
    public int hashCode() {
      return Objects.hash(KeyForBuild.class, label, isBuildPrelude);
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
  static final class KeyForWorkspace extends Key {

    private final Label label;
    private final int workspaceChunk;
    private final RootedPath workspacePath;

    private KeyForWorkspace(Label label, int workspaceChunk, RootedPath workspacePath) {
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
      return keyForWorkspace(loadLabel, workspaceChunk, workspacePath);
    }

    @Override
    BzlCompileValue.Key getCompileKey(Root root) {
      return BzlCompileValue.key(root, label);
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
      if (!(obj instanceof KeyForWorkspace)) {
        return false;
      }
      KeyForWorkspace other = (KeyForWorkspace) obj;
      return label.equals(other.label)
          && workspaceChunk == other.workspaceChunk
          && workspacePath.equals(other.workspacePath);
    }

    @Override
    public int hashCode() {
      return Objects.hash(KeyForWorkspace.class, label, workspaceChunk, workspacePath);
    }
  }

  /**
   * A key for loading a .bzl during {@code @_builtins} evaluation.
   *
   * <p>This kind of key is only requested by {@link StarlarkBuiltinsFunction} and its transitively
   * loaded {@link BzlLoadFunction} calls.
   *
   * <p>The label begins with {@code @_builtins//:}, but there is no actual repo by that name.
   */
  @Immutable
  @AutoCodec.VisibleForSerialization
  static final class KeyForBuiltins extends Key {

    private final Label label;

    private KeyForBuiltins(Label label) {
      this.label = Preconditions.checkNotNull(label);
    }

    @Override
    Label getLabel() {
      return label;
    }

    @Override
    Key getKeyForLoad(Label label) {
      return keyForBuiltins(label);
    }

    @Override
    BzlCompileValue.Key getCompileKey(Root root) {
      return BzlCompileValue.keyForBuiltins(root, label);
    }

    @Override
    public String toString() {
      return label + " (in builtins)";
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof KeyForBuiltins)) {
        return false;
      }
      return this.label.equals(((KeyForBuiltins) obj).label);
    }

    @Override
    public int hashCode() {
      return Objects.hash(KeyForBuiltins.class, label);
    }
  }

  /** Constructs a key for loading a regular (non-workspace) .bzl file, from the .bzl's label. */
  static Key keyForBuild(Label label) {
    return keyInterner.intern(new KeyForBuild(label, /*isBuildPrelude=*/ false));
  }

  /**
   * Constructs a key for loading a .bzl file from the context of evaluating the WORKSPACE file.
   *
   * @param label the label of the bzl file being loaded
   * @param workspaceChunk the workspace chunk that the load statement originated from. If the bzl
   *     file is loaded more than once, this is the chunk that it was first loaded from
   * @param workspacePath the path of the workspace file for the project
   */
  static Key keyForWorkspace(Label label, int workspaceChunk, RootedPath workspacePath) {
    return keyInterner.intern(new KeyForWorkspace(label, workspaceChunk, workspacePath));
  }

  /** Constructs a key for loading a .bzl file within the {@code @_builtins} pseudo-repository. */
  static Key keyForBuiltins(Label label) {
    return keyInterner.intern(new KeyForBuiltins(label));
  }

  /** Constructs a key for loading the special prelude .bzl. */
  static Key keyForBuildPrelude(Label label) {
    return keyInterner.intern(new KeyForBuild(label, /*isBuildPrelude=*/ true));
  }
}
