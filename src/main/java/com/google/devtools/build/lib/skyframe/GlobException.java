// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.io.FileSymlinkInfiniteExpansionException;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.producers.GlobError;
import com.google.devtools.build.skyframe.SkyFunctionException;
import javax.annotation.Nullable;

/**
 * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
 * GlobFunction#compute}.
 */
final class GlobException extends SkyFunctionException {

  GlobException(InconsistentFilesystemException e, Transience transience) {
    super(e, transience);
  }

  GlobException(FileSymlinkInfiniteExpansionException e, Transience transience) {
    super(e, transience);
  }

  /**
   * If any exception are caught and stored in {@link
   * com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState} in {@link
   * GlobFunction}, wrap it inside a {@link GlobException} and throw.
   */
  static void handleExceptions(@Nullable GlobError error) throws GlobException {
    if (error == null) {
      return;
    }
    switch (error.kind()) {
      case INCONSISTENT_FILESYSTEM:
        throw new GlobException(error.inconsistentFilesystem(), Transience.TRANSIENT);
      case FILE_SYMLINK_INFINITE_EXPANSION:
        throw new GlobException(error.fileSymlinkInfiniteExpansion(), Transience.PERSISTENT);
    }
  }
}
