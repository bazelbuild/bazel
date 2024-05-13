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
package com.google.devtools.build.lib.vfs;

import com.google.common.base.Preconditions;
import java.io.IOException;
import javax.annotation.Nullable;

/** An adapter from FileStatus to FileStatusWithDigest. */
public class FileStatusWithDigestAdapter implements FileStatusWithDigest {
  private final FileStatus stat;

  @Nullable
  public static FileStatusWithDigest maybeAdapt(@Nullable FileStatus stat) {
    return stat == null
        ? null
        : stat instanceof FileStatusWithDigest fileStatusWithDigest
            ? fileStatusWithDigest
            : new FileStatusWithDigestAdapter(stat);
  }

  private FileStatusWithDigestAdapter(FileStatus stat) {
    this.stat = Preconditions.checkNotNull(stat);
  }

  @Nullable
  @Override
  public byte[] getDigest() {
    return null;
  }

  @Override
  public boolean isFile() {
    return stat.isFile();
  }

  @Override
  public boolean isSpecialFile() {
    return stat.isSpecialFile();
  }

  @Override
  public boolean isDirectory() {
    return stat.isDirectory();
  }

  @Override
  public boolean isSymbolicLink() {
    return stat.isSymbolicLink();
  }

  @Override
  public long getSize() throws IOException {
    return stat.getSize();
  }

  @Override
  public long getLastModifiedTime() throws IOException {
    return stat.getLastModifiedTime();
  }

  @Override
  public long getLastChangeTime() throws IOException {
    return stat.getLastChangeTime();
  }

  @Override
  public long getNodeId() throws IOException {
    return stat.getNodeId();
  }
}
