// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.errorprone.annotations.CheckReturnValue;

/**
 * Interface that provides details on how UnixGlob should discriminate between different Paths.
 * Instances of this interface will be handed either real directories or real files after symlink
 * resolution and excluding special files.
 */
@CheckReturnValue
public interface UnixGlobPathDiscriminator {

  /**
   * Determine if UnixGlob should enumerate entries in this directory and traverse it during
   * recursive globbing. Defaults to true.
   */
  default boolean shouldTraverseDirectory(Path path) {
    return true;
  }

  /**
   * Determine if UnixGlob should include the given path in a {@code List<Path>} result. Defaults to
   * true;
   */
  default boolean shouldIncludePathInResult(Path path, boolean isDirectory) {
    return true;
  }
}
