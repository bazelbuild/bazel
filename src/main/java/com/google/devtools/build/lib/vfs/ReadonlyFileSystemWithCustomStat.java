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

import java.io.IOException;
import java.io.OutputStream;

/** Functionally like a read-only {@link AbstractFileSystemWithCustomStat}. */
public abstract class ReadonlyFileSystemWithCustomStat extends AbstractFileSystemWithCustomStat {
  public ReadonlyFileSystemWithCustomStat(DigestHashFunction hashFunction) {
    super(hashFunction);
  }

  protected IOException modificationException() {
    String longname = this.getClass().getName();
    String shortname = longname.substring(longname.lastIndexOf('.') + 1);
    return new IOException(
        shortname + " does not support mutating operations");
  }

  @Override
  protected OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
      throws IOException {
    throw modificationException();
  }

  @Override
  protected void setReadable(PathFragment path, boolean readable) throws IOException {
    throw modificationException();
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    throw modificationException();
  }

  @Override
  protected void setExecutable(PathFragment path, boolean executable) {
    throw new UnsupportedOperationException("setExecutable");
  }

  @Override
  public boolean supportsModifications(PathFragment path) {
    return false;
  }

  @Override
  public boolean supportsSymbolicLinksNatively(PathFragment path) {
    return false;
  }

  @Override
  public boolean supportsHardLinksNatively(PathFragment path) {
    return false;
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return true;
  }

  @Override
  public boolean createDirectory(PathFragment path) throws IOException {
    throw modificationException();
  }

  @Override
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    throw modificationException();
  }

  @Override
  protected void createSymbolicLink(PathFragment linkPath, PathFragment targetFragment)
      throws IOException {
    throw modificationException();
  }

  @Override
  protected void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    throw modificationException();
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    throw modificationException();
  }

  @Override
  protected boolean delete(PathFragment path) throws IOException {
    throw modificationException();
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    throw modificationException();
  }
}

