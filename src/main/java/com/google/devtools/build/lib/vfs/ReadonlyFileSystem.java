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

/**
 * An abstract partial implementation of FileSystem for read-only implementations.
 *
 * <p>Any ReadonlyFileSystem does not support the following:
 *
 * <ul>
 *   <li>{@link #createDirectory(LocalPath)}
 *   <li>{@link #createSymbolicLink(LocalPath, String)}
 *   <li>{@link #delete(LocalPath)}
 *   <li>{@link #getOutputStream(LocalPath)}
 *   <li>{@link #renameTo(LocalPath, LocalPath)}
 *   <li>{@link #setExecutable(LocalPath, boolean)}
 *   <li>{@link #setLastModifiedTime(LocalPath, long)}
 *   <li>{@link #setWritable(LocalPath, boolean)}
 * </ul>
 *
 * The above calls will always result in an {@link IOException}.
 */
public abstract class ReadonlyFileSystem extends AbstractFileSystem {

  protected ReadonlyFileSystem() {
  }

  protected IOException modificationException() {
    String longname = this.getClass().getName();
    String shortname = longname.substring(longname.lastIndexOf('.') + 1);
    return new IOException(
        shortname + " does not support mutating operations");
  }

  @Override
  protected OutputStream getOutputStream(LocalPath path, boolean append) throws IOException {
    throw modificationException();
  }

  @Override
  protected void setReadable(LocalPath path, boolean readable) throws IOException {
    throw modificationException();
  }

  @Override
  public void setWritable(LocalPath path, boolean writable) throws IOException {
    throw modificationException();
  }

  @Override
  protected void setExecutable(LocalPath path, boolean executable) {
    throw new UnsupportedOperationException("setExecutable");
  }

  @Override
  public boolean supportsModifications(LocalPath path) {
    return false;
  }

  @Override
  public boolean supportsSymbolicLinksNatively(LocalPath path) {
    return false;
  }

  @Override
  public boolean supportsHardLinksNatively(LocalPath path) {
    return false;
  }

  @Override
  public boolean isFilePathCaseSensitive() {
    return true;
  }

  @Override
  public boolean createDirectory(LocalPath path) throws IOException {
    throw modificationException();
  }

  @Override
  protected void createSymbolicLink(LocalPath linkPath, String targetFragment) throws IOException {
    throw modificationException();
  }

  @Override
  public void renameTo(LocalPath sourcePath, LocalPath targetPath) throws IOException {
    throw modificationException();
  }

  @Override
  public boolean delete(LocalPath path) throws IOException {
    throw modificationException();
  }

  @Override
  public void setLastModifiedTime(LocalPath path, long newTime) throws IOException {
    throw modificationException();
  }

  @Override
  protected void createFSDependentHardLink(LocalPath linkPath, LocalPath originalPath)
      throws IOException {
    throw modificationException();
  }
}
