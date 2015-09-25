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
package com.google.devtools.build.lib.vfs.inmemoryfs;

import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.ScopeEscapableFileSystem;

/**
 * Interface definition for a file status that may signify that the
 * referenced path falls outside the scope of the file system (see
 * {@link ScopeEscapableFileSystem}) and can provide the "escaped"
 * version of that path suitable for re-delegation to another file
 * system.
 */
interface ScopeEscapableStatus extends FileStatus {

  /**
   * Returns true if this status corresponds to a path that leaves
   * the file system's scope, false otherwise.
   */
  boolean outOfScope();

  /**
   * If this status represents a path that leaves the file system's scope,
   * returns the requested path resolved up to the point where it first
   * escapes the file system. For example: if the file system is mapped to
   * /foo, the requested path is /foo/link1/link2/link3, and link1 -> /bar,
   * this returns /bar/link2/link3.
   *
   * <p>If this status doesn't represent a scope-escaping path, returns
   * null.
   */
  PathFragment getEscapingPath();
}
