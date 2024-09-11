// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.config;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/** Key for {@link PlatformMappingValue} based on the location of the mapping file. */
@ThreadSafety.Immutable
@AutoCodec
public final class PlatformMappingKey implements SkyKey {

  private static final SkyKeyInterner<PlatformMappingKey> interner = SkyKey.newInterner();

  /** Default key to use when the user does not explicitly set {@code --platform_mappings}. */
  public static final PlatformMappingKey DEFAULT =
      create(PathFragment.create("platform_mappings"), /* wasExplicitlySetByUser= */ false);

  /**
   * Creates a platform mapping key with the given, main workspace-relative path to the mappings
   * file which was specified by the user via the {@code --platform_mappings} flag.
   */
  public static PlatformMappingKey createExplicitlySet(PathFragment workspaceRelativeMappingPath) {
    return create(workspaceRelativeMappingPath, /* wasExplicitlySetByUser= */ true);
  }

  private static PlatformMappingKey create(
      PathFragment workspaceRelativeMappingPath, boolean wasExplicitlySetByUser) {
    return interner.intern(
        new PlatformMappingKey(workspaceRelativeMappingPath, wasExplicitlySetByUser));
  }

  @VisibleForSerialization
  @AutoCodec.Interner
  static PlatformMappingKey intern(PlatformMappingKey key) {
    return interner.intern(key);
  }

  private final PathFragment path;
  private final boolean wasExplicitlySetByUser;

  private PlatformMappingKey(PathFragment path, boolean wasExplicitlySetByUser) {
    this.path = path;
    this.wasExplicitlySetByUser = wasExplicitlySetByUser;
  }

  /** Returns the main-workspace relative path this mapping's mapping file can be found at. */
  public PathFragment getWorkspaceRelativeMappingPath() {
    return path;
  }

  boolean wasExplicitlySetByUser() {
    return wasExplicitlySetByUser;
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.PLATFORM_MAPPING;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof PlatformMappingKey that)) {
      return false;
    }
    return path.equals(that.path) && wasExplicitlySetByUser == that.wasExplicitlySetByUser;
  }

  @Override
  public int hashCode() {
    return path.hashCode() * 31 + Boolean.hashCode(wasExplicitlySetByUser);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("path", path)
        .add("wasExplicitlySetByUser", wasExplicitlySetByUser)
        .toString();
  }

  @Override
  public SkyKeyInterner<PlatformMappingKey> getSkyKeyInterner() {
    return interner;
  }
}
