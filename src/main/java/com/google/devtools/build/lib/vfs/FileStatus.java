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

/**
 * File status: mode, mtime, size, etc.
 *
 * <p>The result of calling any {@code FileStatus} instance method is not
 * guaranteed to result in I/O to the file system at the moment of the call.
 * The I/O providing the result (and hence the throwing of an I/O exception,
 * where applicable) may occur at any moment between the call to {@link
 * FileSystem#stat} and the call of the {@code FileStatus} instance method.
 *
 * <p>Callers therefore cannot assume that all the values are populated
 * atomically, or that the results of any two {@code FileStatus} methods
 * correspond to state of the file system at a single moment in time.  Nor may
 * they assume that repeated successful calls to any method of the same
 * instance will return the same value.
 *
 * <p>(This permits conforming implementations to use an atomic {@code stat(2)}
 * call on file systems where it is available, and individual accessor methods
 * on those where it is not.  Caching is possible but not required.)
 */
public interface FileStatus {

  /**
   * Returns true iff this file is a regular file or {@code isSpecial()}.
   */
  boolean isFile();

  /**
   * Returns true iff this file is a directory.
   */
  boolean isDirectory();

  /**
   * Returns true iff this file is a symbolic link.
   */
  boolean isSymbolicLink();

  /**
   * Returns true iff this file is a special file (e.g. socket, fifo or device). {@link #getSize()}
   * can't be trusted for such files.
   */
  boolean isSpecialFile();

  /**
   * Returns the total size, in bytes, of this file.
   */
  long getSize() throws IOException;

  /**
   * Returns the last modified time of this file's data (milliseconds since
   * UNIX epoch).
   *
   * TODO(bazel-team): Unix actually gives us nanosecond resolution for mtime and ctime. Consider
   * making use of this.
   */
  long getLastModifiedTime() throws IOException;

  /**
   * Returns the last change time of this file, where change means any change
   * to the file, including metadata changes (milliseconds since UNIX epoch).
   */
  long getLastChangeTime() throws IOException;

  /**
   * Returns the unique file node id. Usually it is computed using both device
   * and inode numbers.
   *
   * <p>Think of this value as a reference to the underlying inode. "mv"ing file a to file b
   * ought to cause the node ID of b to change, but appending / modifying b should not.
   */
  long getNodeId() throws IOException;

  /**
   * Returns the file's permissions in POSIX format (e.g. 0755) if possible without performing
   * additional IO, otherwise (or if unsupported by the file system) returns -1.
   *
   * <p>If accurate group and other permissions aren't available, the returned value should attempt
   * to mimic a umask of 022 (i.e. read and execute permissions extend to group and other, write
   * does not).
   */
  default int getPermissions() {
    return -1;
  }
}
