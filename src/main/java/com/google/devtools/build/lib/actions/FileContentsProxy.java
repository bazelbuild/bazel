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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileStatus;
import java.io.IOException;
import java.util.Objects;

/**
 * In case we can't get a fast digest from the filesystem, we store this metadata as a proxy to the
 * file contents. Currently it is two timestamps and a "node id". On Linux we use both ctime and
 * mtime and inode number. We might want to add the device number in the future.
 *
 * <p>For a Linux example of why mtime alone is insufficient, note that 'mv' preserves mtime. So if
 * files 'a' and 'b' initially have the same timestamp, then we would think 'b' is unchanged after
 * the user executes `mv a b` between two builds.
 *
 * <p>On Linux we also need mtime for hardlinking sandbox, since updating the inode reference
 * counter preserves mtime, but updates ctime. isModified() call can be used to compare two
 * FileContentsProxys of hardlinked files.
 */
public final class FileContentsProxy {
  private final long ctime;
  private final long mtime;
  private final long nodeId;

  public FileContentsProxy(long ctime, long mtime, long nodeId) {
    this.ctime = ctime;
    this.mtime = mtime;
    this.nodeId = nodeId;
  }

  public static FileContentsProxy create(FileStatus stat) throws IOException {
    // Note: there are file systems that return mtime for this call instead of ctime, such as the
    // WindowsFileSystem.
    return new FileContentsProxy(
        stat.getLastChangeTime(), stat.getLastModifiedTime(), stat.getNodeId());
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    }

    if (!(other instanceof FileContentsProxy that)) {
      return false;
    }

    return ctime == that.ctime && mtime == that.mtime && nodeId == that.nodeId;
  }

  /**
   * Can be used when hardlink reference counter changes should not be considered a file
   * modification. Is only comparing mtime and not ctime and is therefore not detecting changed
   * metadata like permission.
   */
  @SuppressWarnings("ReferenceEquality")
  public boolean isModified(FileContentsProxy other) {
    if (other == this) {
      return false;
    }
    // true if nodeId are different or inode has a new mtime
    return nodeId != other.nodeId || mtime != other.mtime;
  }

  @Override
  public int hashCode() {
    return Objects.hash(ctime, mtime, nodeId);
  }

  void addToFingerprint(Fingerprint fp) {
    fp.addLong(ctime);
    fp.addLong(mtime);
    fp.addLong(nodeId);
  }

  @Override
  public String toString() {
    return prettyPrint();
  }

  public String prettyPrint() {
    return String.format("ctime of %d and mtime of %d and nodeId of %d", ctime, mtime, nodeId);
  }
}
