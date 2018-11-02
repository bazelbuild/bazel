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
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * A value that represents a Skylark import lookup result. The lookup value corresponds to exactly
 * one Skylark file, identified by an absolute {@link Label} {@link SkyKey} argument. The Label
 * should not reference the special {@code external} package.
 */
@AutoCodec
public class SkylarkImportLookupValue implements SkyValue {

  private final Extension environmentExtension;
  /**
   * The immediate Skylark file dependency descriptor class corresponding to this value.
   * Using this reference it's possible to reach the transitive closure of Skylark files
   * on which this Skylark file depends.
   */
  private final SkylarkFileDependency dependency;

  @VisibleForTesting
  public SkylarkImportLookupValue(
      Extension environmentExtension, SkylarkFileDependency dependency) {
    this.environmentExtension = Preconditions.checkNotNull(environmentExtension);
    this.dependency = Preconditions.checkNotNull(dependency);
  }

  /**
   * Returns the Extension
   */
  public Extension getEnvironmentExtension() {
    return environmentExtension;
  }

  /**
   * Returns the immediate Skylark file dependency corresponding to this import lookup value.
   */
  public SkylarkFileDependency getDependency() {
    return dependency;
  }

  /**
   * SkyKey for a Skylark import composed of the label of the Skylark extension and wether it is
   * loaded from the WORKSPACE file or from a BUILD file.
   */
  @Immutable
  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static final class SkylarkImportLookupKey implements SkyKey {
    private static final Interner<SkylarkImportLookupKey> interner =
        BlazeInterners.newWeakInterner();

    public final Label importLabel;
    public final boolean inWorkspace;
    // a workspaceChunk = -1 means inWorkspace is false
    public final int workspaceChunk;
    // a null rooted workspace path means inWorkspace is false
    public final RootedPath workspacePath;

    private SkylarkImportLookupKey(Label importLabel, boolean inWorkspace, int workspaceChunk, RootedPath workspacePath) {
      Preconditions.checkNotNull(importLabel);
      Preconditions.checkArgument(!importLabel.getPackageIdentifier().getRepository().isDefault());
      this.importLabel = importLabel;
      this.inWorkspace = inWorkspace;
      this.workspaceChunk = workspaceChunk;
      this.workspacePath = workspacePath;
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static SkylarkImportLookupKey create(Label importLabel, boolean inWorkspace, int workspaceChunk, RootedPath workspacePath) {
      return interner.intern(new SkylarkImportLookupKey(importLabel, inWorkspace, workspaceChunk, workspacePath));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.SKYLARK_IMPORTS_LOOKUP;
    }

    @Override
    public String toString() {
      return importLabel + (inWorkspace ? " (in workspace)" : "");
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof SkylarkImportLookupKey)) {
        return false;
      }
      SkylarkImportLookupKey other = (SkylarkImportLookupKey) obj;
      return importLabel.equals(other.importLabel)
          && inWorkspace == other.inWorkspace
          && workspaceChunk == other.workspaceChunk
          // note this might break if workspacePath can be nullable
          // actually, can skykeys contain null values??
          && Objects.equals(workspacePath, other.workspacePath);
    }

    @Override
    public int hashCode() {
      return Objects.hash(importLabel, inWorkspace, workspaceChunk, workspacePath);
    }
  }

  static SkyKey key(Label importLabel, boolean inWorkspace, int workspaceChunk, RootedPath workspacePath) {
    return SkylarkImportLookupKey.create(importLabel, inWorkspace, workspaceChunk, workspacePath);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof SkylarkImportLookupValue)) {
      return false;
    }
    SkylarkImportLookupValue other = (SkylarkImportLookupValue) obj;
    return environmentExtension.equals(other.getEnvironmentExtension())
        && dependency.equals(other.getDependency());
  }

  @Override
  public int hashCode() {
    return Objects.hash(environmentExtension, dependency);
  }
}
