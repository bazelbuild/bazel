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
 * An IOException subclass that is thrown when a file system access is denied. The message is
 * generally "Permission denied".
 */
public class FileAccessException extends IOException {
  /**
   * Constructs a <code>FileAccessException</code> with <code>null</code>
   * as its error detail message.
   */
  public FileAccessException() {
    super();
  }

  /**
   * Constructs an <code>FileAccessException</code> with the specified detail
   * message. The error message string <code>s</code> can later be
   * retrieved by the <code>{@link java.lang.Throwable#getMessage}</code>
   * method of class <code>java.lang.Throwable</code>.
   *
   * @param s the detail message.
   */
  public FileAccessException(String s) {
    super(s);
  }
}
