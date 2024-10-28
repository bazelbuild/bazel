// Copyright 2024 The Bazel Authors. All rights reserved.
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

/**
 * An interface implemented by file systems that support pinning files to a read-only snapshot.
 *
 * <p>This is used by {@link SymlinkTreeHelper} to obtain the underlying non-snapshotting file
 * system, so that symlink trees are not pinned to a read-only snapshot and reflect changes made
 * after the symlink tree was built.
 */
public interface SnapshottingFileSystem {
  /** Returns the underlying non-snapshotting file system. */
  FileSystem getUnderlyingNonSnapshottingFileSystem();
}
