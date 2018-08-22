// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.vfs.DigestHashFunction.DefaultHashFunctionNotSetException;
import java.io.IOException;

/**
 * An {@link AbstractFileSystem} that provides default implementations of {@link FileSystem#isFile}
 * et al. in terms of {@link FileSystem#stat} (rather than the other way around, which is what
 * {@link FileSystem} does).
 */
public abstract class AbstractFileSystemWithCustomStat extends AbstractFileSystem {

  public AbstractFileSystemWithCustomStat() throws DefaultHashFunctionNotSetException {}

  public AbstractFileSystemWithCustomStat(DigestHashFunction hashFunction) {
    super(hashFunction);
  }

  @Override
  protected boolean isFile(Path path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null ? stat.isFile() : false;
  }

  @Override
  protected boolean isSpecialFile(Path path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null ? stat.isSpecialFile() : false;
  }

  @Override
  protected boolean isSymbolicLink(Path path) {
    FileStatus stat = statNullable(path, false);
    return stat != null ? stat.isSymbolicLink() : false;
  }

  @Override
  protected boolean isDirectory(Path path, boolean followSymlinks) {
    FileStatus stat = statNullable(path, followSymlinks);
    return stat != null ? stat.isDirectory() : false;
  }

  @Override
  protected abstract FileStatus stat(Path path, boolean followSymlinks) throws IOException;
}

