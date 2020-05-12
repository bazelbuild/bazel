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

  /**
   * SkyKey for a Starlark import composed of the label of the Starlark extension and whether it is
   * loaded from the WORKSPACE file or from a BUILD file.
   */
  @Immutable
  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static final class Key implements SkyKey {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    public final Label importLabel;
    public final boolean inWorkspace;
    // a workspaceChunk = -1 means inWorkspace is false
    public final int workspaceChunk;
    // a null rooted workspace path means inWorkspace is false
    public final RootedPath workspacePath;

    private Key(
        Label importLabel, boolean inWorkspace, int workspaceChunk, RootedPath workspacePath) {
      Preconditions.checkNotNull(importLabel);
      Preconditions.checkArgument(!importLabel.getPackageIdentifier().getRepository().isDefault());
      this.importLabel = importLabel;
      this.inWorkspace = inWorkspace;
      this.workspaceChunk = workspaceChunk;
      this.workspacePath = workspacePath;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(
        Label importLabel, boolean inWorkspace, int workspaceChunk, RootedPath workspacePath) {
      return interner.intern(new Key(importLabel, inWorkspace, workspaceChunk, workspacePath));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.STARLARK_IMPORTS_LOOKUP;
    }

    @Override
    public String toString() {
      return importLabel + (inWorkspace ? " (in workspace)" : "");
    }

    Label getImportLabel() {
      return importLabel;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Key)) {
        return false;
      }
      Key other = (Key) obj;
      return importLabel.equals(other.importLabel)
          && inWorkspace == other.inWorkspace
          && workspaceChunk == other.workspaceChunk
          && Objects.equals(workspacePath, other.workspacePath);
    }

    @Override
    public int hashCode() {
      return Objects.hash(importLabel, inWorkspace, workspaceChunk, workspacePath);
    }
  }

  /**
   * Creates a {@link StarlarkImportLookupValue.Key}.
   *
   * @param importLabel the label of the bzl file being loaded
   * @param workspaceChunk the workspace chunk that the load statement originated from. If the bzl
   *     file is loaded more than once, this is the chunk that it was first loaded from
   * @param workspacePath the path of the workspace file for the project
   */
  static Key keyInWorkspace(Label importLabel, int workspaceChunk, RootedPath workspacePath) {
    return Key.create(importLabel, /* inWorkspace= */ true, workspaceChunk, workspacePath);
  }

  /**
   * Convenience method to construct a key for load statements that do not originate from a
   * workspace file.
   *
   * @param importLabel the label of the bzl file being loaded
   */
  static Key key(Label importLabel) {
    return Key.create(
        importLabel, /* inWorkspace= */ false, /* workspaceChunk= */ -1, /* workspacePath= */ null);
  }
}
