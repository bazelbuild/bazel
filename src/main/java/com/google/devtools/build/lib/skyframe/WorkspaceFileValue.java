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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Objects;

/**
 * A SkyValue that contains the result of the parsing of one part of the WORKSPACE file. The parsing
 * of the WORKSPACE file is split before each series of load statement because we need to resolve
 * repositories before being able to load from those repositories.
 */
public class WorkspaceFileValue implements SkyValue {

  /**
   * Argument for the SkyKey to request a WorkspaceFileValue.
   */
  @Immutable
  public static class WorkspaceFileKey {
    private final RootedPath path;
    private final int idx;

    /**
     * Creates a Key for the WorkspaceFileFunction. The path to the workspace file is specified
     * by {@code path}. This key will ask WorkspaceFileFunction to get the {@code idx+1}-th part of
     * the workspace file (so idx = 0 represents the first part, idx = 1, the second part, etc...).
     */
    public WorkspaceFileKey(RootedPath path, int idx) {
      this.path = path;
      this.idx = idx;
    }

    public RootedPath getPath() {
      return path;
    }

    public int getIndex() {
      return idx;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof WorkspaceFileKey)) {
        return false;
      }
      WorkspaceFileKey other = (WorkspaceFileKey) obj;
      return Objects.equals(path, other.path) && idx == other.idx;
    }

    @Override
    public int hashCode() {
      return Objects.hash(path.hashCode(), idx);
    }

    @Override
    public String toString() {
      return path + ", " + idx;
    }
  }

  private final Package pkg;
  private final int idx;
  private final RootedPath path;
  private final boolean hasNext;

  public WorkspaceFileValue(Package pkg, RootedPath path, int idx, boolean hasNext) {
    this.pkg = Preconditions.checkNotNull(pkg);
    this.idx = idx;
    this.path = path;
    this.hasNext = hasNext;
  }

  /**
   * Returns the package. This package may contain errors, in which case the caller should throw
   * a {@link BuildFileContainsErrorsException}.
   */
  public Package getPackage() {
    return pkg;
  }

  @Override
  public String toString() {
    return "<WorkspaceFileValue idx=" + idx + ">";
  }

  private static SkyKey key(RootedPath path, int idx) {
    return new SkyKey(SkyFunctions.WORKSPACE_FILE, new WorkspaceFileKey(path, idx));
  }

  public static SkyKey key(RootedPath path) {
    return key(path, 0);
  }

  /**
   * Get the key for the next WorkspaceFileValue or null if this value is the last part of the
   * workspace file.
   */
  public SkyKey next() {
    if (hasNext) {
      return key(path, idx + 1);
    } else {
      return null;
    }
  }

  /**
   * The workspace file parsing is cut in several parts and this function returns the index of the
   * part of the workspace file that this value holds. For the first part, this index will be 0, for
   * the second part, it will be 1 and so on.
   */
  public int getIndex() {
    return idx;
  }

  /**
   * The workspace file parsing is cut in several parts and this function returns true if there is
   * a part following the part holds by this value (or false if this is the last part of the
   * WORKSPACE file.
   *
   * <p>This method is public for serialization of the WorkspaceFileValue, #next() should be used
   * to iterate instead of this method.
   */
  public boolean hasNext() {
    return hasNext;
  }

  public RootedPath getPath() {
    return path;
  }
}
