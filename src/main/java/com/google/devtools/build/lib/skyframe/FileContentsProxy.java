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

import com.google.devtools.build.lib.vfs.FileStatus;
import java.io.IOException;
import java.io.Serializable;
import java.util.Objects;

/**
 * In case we can't get a fast digest from the filesystem, we store this metadata as a proxy to
 * the file contents. Currently it is a pair of the mtime and "value id" (which is right now just
 * the inode number). We may wish to augment this object with the following data:
 * a. the device number
 * b. the ctime, which cannot be tampered with in userspace
 *
 * <p>For an example of why mtime alone is insufficient, note that 'mv' preserves timestamps. So if
 * files 'a' and 'b' initially have the same timestamp, then we would think 'b' is unchanged after
 * the user executes `mv a b` between two builds.
 */
public final class FileContentsProxy implements Serializable {
  private final long mtime;
  private final long valueId;

  /**
   * Visible for serialization / deserialization. Do not use this method, but call {@link #create}
   * instead.
   */
  public FileContentsProxy(long mtime, long valueId) {
    this.mtime = mtime;
    this.valueId = valueId;
  }

  public static FileContentsProxy create(FileStatus stat) throws IOException {
    return new FileContentsProxy(stat.getLastModifiedTime(), stat.getNodeId());
  }

  public long getMtime() {
    return mtime;
  }

  public long getValueId() {
    return valueId;
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    }

    if (!(other instanceof FileContentsProxy)) {
      return false;
    }

    FileContentsProxy that = (FileContentsProxy) other;
    return mtime == that.mtime && valueId == that.valueId;
  }

  @Override
  public int hashCode() {
    return Objects.hash(mtime, valueId);
  }

  @Override
  public String toString() {
    return prettyPrint();
  }

  public String prettyPrint() {
    return String.format("mtime of %d and valueId of %d", mtime, valueId);
  }
}

