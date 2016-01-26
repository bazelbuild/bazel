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
 * An abstract partial implementation of FileSystem for read-only
 * implementations.
 *
 * <p>Any ReadonlyFileSystem does not support the following:
 * <ul>
 * <li>{@link #createDirectory(Path)}</li>
 * <li>{@link #createSymbolicLink(Path, PathFragment)}</li>
 * <li>{@link #delete(Path)}</li>
 * <li>{@link #getOutputStream(Path)}</li>
 * <li>{@link #renameTo(Path, Path)}</li>
 * <li>{@link #setExecutable(Path, boolean)}</li>
 * <li>{@link #setLastModifiedTime(Path, long)}</li>
 * <li>{@link #setWritable(Path, boolean)}</li>
 * </ul>
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
  protected OutputStream getOutputStream(Path path, boolean append) throws IOException {
    throw modificationException();
  }

  @Override
  protected void setReadable(Path path, boolean readable) throws IOException {
    throw modificationException();
  }

  @Override
  protected void setWritable(Path path, boolean writable) throws IOException {
    throw modificationException();
  }

  @Override
  protected void setExecutable(Path path, boolean executable) {
    throw new UnsupportedOperationException("setExecutable");
  }

  @Override
  public boolean supportsModifications() {
    return false;
  }

  @Override
  public boolean supportsSymbolicLinksNatively() {
    return false;
  }

  @Override
  protected boolean createDirectory(Path path) throws IOException {
    throw modificationException();
  }

  @Override
  protected void createSymbolicLink(Path linkPath, PathFragment targetFragment) throws IOException {
    throw modificationException();
  }

  @Override
  protected void renameTo(Path sourcePath, Path targetPath) throws IOException {
    throw modificationException();
  }

  @Override
  protected boolean delete(Path path) throws IOException {
    throw modificationException();
  }

  @Override
  protected void setLastModifiedTime(Path path, long newTime) throws IOException {
    throw modificationException();
  }

}
