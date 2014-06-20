// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.view.fileset;

import java.io.IOException;

/**
 * This exception is thrown by FilesetUtil methods that try to resolve symlinks
 * when they find a dangling one.
 */
class DanglingSymlinkException extends IOException {
  final String path;

  public DanglingSymlinkException(String path) {
    super("Dangling symlink: " + path);
    this.path = path;
  }

  public String getPath() {
    return path;
  }
}
