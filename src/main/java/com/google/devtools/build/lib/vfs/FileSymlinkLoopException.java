// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.io.FileSymlinkException;

/** A {@link FileSymlinkException} that indicates a symlink loop. */
public final class FileSymlinkLoopException extends FileSymlinkException {
  FileSymlinkLoopException(String message) {
    super(message);
  }

  public FileSymlinkLoopException(PathFragment pathFragment) {
    this(pathFragment.getPathString() + " (Too many levels of symbolic links)");
  }

  @Override
  public String getUserFriendlyMessage() {
    return getMessage();
  }
}
