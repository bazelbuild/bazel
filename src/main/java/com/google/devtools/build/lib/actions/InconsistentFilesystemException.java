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
package com.google.devtools.build.lib.actions;

import java.io.IOException;

/**
 * Used to indicate a filesystem inconsistency, e.g. file 'a/b' exists but directory 'a' doesn't
 * exist. This generally means the result of the build is undefined but we shouldn't crash hard.
 */
public class InconsistentFilesystemException extends IOException {
  public InconsistentFilesystemException(String inconsistencyMessage) {
    super(
        "Inconsistent filesystem operations. "
            + inconsistencyMessage
            + " The results of the "
            + "build are not guaranteed to be correct. You should probably run 'bazel clean' and "
            + "investigate the filesystem inconsistency (likely due to filesystem updates "
            + "concurrent with the build)");
  }
}
