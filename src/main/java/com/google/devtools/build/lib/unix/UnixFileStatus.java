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
package com.google.devtools.build.lib.unix;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;

/**
 * An implementation of {@link FileStatus} backed by the result of a stat(2) system call.
 *
 * <p>This class is optimized for memory usage. Fields not required by Bazel are omitted.
 */
public final class UnixFileStatus implements FileStatus {

  private final int mode;
  private final long mtime; // milliseconds since Unix epoch
  private final long ctime; // milliseconds since Unix epoch
  private final long size;
  private final long ino;

  /** Constructs a {@link UnixFileStatus} from a {@link NativePosixFilesService.Stat}. */
  UnixFileStatus(NativePosixFilesService.Stat stat) {
    this.mode = stat.mode();
    this.mtime = stat.mtime();
    this.ctime = stat.ctime();
    this.size = stat.size();
    this.ino = stat.ino();
  }

  public Dirent.Type getDirentType() {
    return UnixMode.getDirentTypeFromMode(mode);
  }

  @Override
  public long getNodeId() {
    // TODO(tjgq): Consider deriving this value from both st_dev and st_ino.
    return ino;
  }

  @Override
  public boolean isFile() {
    return UnixMode.isFile(mode);
  }

  @Override
  public boolean isSpecialFile() {
    return UnixMode.isSpecialFile(mode);
  }

  @Override
  public boolean isDirectory() {
    return UnixMode.isDirectory(mode);
  }

  @Override
  public boolean isSymbolicLink() {
    return UnixMode.isSymbolicLink(mode);
  }

  @Override
  public int getPermissions() {
    return UnixMode.getPermissions(mode);
  }

  @Override
  public long getSize() {
    return size;
  }

  @Override
  public long getLastModifiedTime() {
    return mtime;
  }

  @Override
  public long getLastChangeTime() {
    return ctime;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("mode", String.format("0%06o", mode))
        .add("mtime", mtime)
        .add("ctime", ctime)
        .add("size", size)
        .add("ino", ino)
        .toString();
  }
}
