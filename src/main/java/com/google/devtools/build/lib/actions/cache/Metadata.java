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

/**
 * An interface to represent the state of a file (or directory). This is used to determine whether a
 * file has changed or not.
 *
 * <p>NB! Several other parts of Blaze are relying on the fact that metadata uses mtime and not
 * ctime. If metadata is ever changed to use ctime, all uses of Metadata must be carefully examined.
 */
public interface Metadata {
  /**
   * Marker interface for singleton implementations of the Metadata interface. This is only needed
   * for a correct implementation of {@code equals}.
   */
  public interface Singleton {
  }

  /**
   * Whether the underlying file system object is a file or a symlink to a file, rather than a
   * directory. All files are guaranteed to have a digest, and {@link #getDigest} must only be
   * called on files.
   */
  boolean isFile();

  /**
   * Returns the file's digest; must only be called on objects for which {@link #isFile} returns
   * true.
   *
   * <p>The return value is owned by the cache and must not be modified.
   */
  byte[] getDigest();

  /** Returns the file's size, or 0 if the underlying file system object is not a file. */
  // TODO(ulfjack): Throw an exception if it's not a file.
  long getSize();

  /**
   * Returns the last modified time; must only be called on objects for which {@link #isFile}
   * returns false.
   */
  long getModifiedTime();
}
