// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.vfs.RootedPath;

import java.io.IOException;

/**
 * <p>This is an implementation detail of {@link FileFunction} to to evaluate a symlink linking to
 * file outside of known/allowed directory structures.
 *
 * <p>Extends {@link IOException} to ensure it is properly handled by consumers of
 * {@link FileValue}.
 */
class SymlinkOutsidePackageRootsException extends IOException {
  /**
   * @param symlinkPath the {@link RootedPath} that links to an outside path.
   * @param outsidePath the {@link RootedPath} that triggered this exception.
   */
  public SymlinkOutsidePackageRootsException(RootedPath symlinkPath, RootedPath outsidePath) {
    super("Encountered symlink " + symlinkPath + " linking to external mutable " + outsidePath);
  }
}
