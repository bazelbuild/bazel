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
package com.google.devtools.build.lib.actions.cache;

import com.google.devtools.build.lib.vfs.FileStatus;

/**
 * A FileStatus corresponding to a file that is not determined by querying the file system.
 *
 * <p>Do not use this in combination with MetadataHandler or FileContentsProxy! FileContentsProxy
 * may use ctime, which this class does not support.
 */
public class InjectedStat implements FileStatus {

  private final long mtime;
  private final long size;
  private final long nodeId;

  public InjectedStat(long mtime, long size, long nodeId) {
    this.mtime = mtime;
    this.size = size;
    this.nodeId = nodeId;
  }

  @Override
  public boolean isFile() {
    return true;
  }

  @Override
  public boolean isSpecialFile() {
    return false;
  }

  @Override
  public boolean isDirectory() {
    return false;
  }

  @Override
  public boolean isSymbolicLink() {
    return false;
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
    return getLastModifiedTime();
  }

  @Override
  public long getNodeId() {
    return nodeId;
  }
}
