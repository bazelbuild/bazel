// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collections;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * Refresh roots value, computed by the global refresh() function call in WORKSPACE file header.
 * {@see WorkspaceFactory#refresh}
 */
@AutoCodec
public class RefreshRootsValue implements SkyValue {
  private final NavigableMap<PathFragment, RepositoryName> roots;

  @AutoCodec.VisibleForSerialization @AutoCodec
  static final SkyKey REFRESH_ROOTS_KEY = () -> SkyFunctions.REFRESH_ROOTS;

  public RefreshRootsValue(NavigableMap<PathFragment, RepositoryName> roots) {
    this.roots = roots;
  }

  public static SkyKey key() {
    return REFRESH_ROOTS_KEY;
  }

  public Map<PathFragment, RepositoryName> getRoots() {
    return Collections.unmodifiableMap(roots);
  }

  public boolean isEmpty() {
    return roots.isEmpty();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    RefreshRootsValue that = (RefreshRootsValue) o;
    return roots.equals(that.roots);
  }

  @Override
  public int hashCode() {
    return Objects.hash(roots);
  }

  @Nullable
  static RepositoryName getRepositoryForRefreshRoot(
      final RefreshRootsValue refreshRootsValue,
      final RootedPath rootedPath) {

    PathFragment relativePath = rootedPath.getRootRelativePath();
    Map.Entry<PathFragment, RepositoryName> entry = refreshRootsValue.roots
        .floorEntry(relativePath);
    if (entry != null) {
      return entry.getValue();
    }
    return null;
  }
}
