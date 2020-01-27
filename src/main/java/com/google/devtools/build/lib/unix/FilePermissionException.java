// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.unix;

import java.io.IOException;

/**
 * An IOException subclass that is thrown when a POSIX filesystem call returns
 * an EPERM errno. The message is generally "Operation not permitted".
 */
public class FilePermissionException extends IOException {
  /**
   * Constructs a <code>FilePermissionException</code> with <code>null</code>
   * as its error detail message.
   */
  public FilePermissionException() {
    super();
  }

  /**
   * Constructs an <code>FilePermissionException</code> with the specified detail
   * message. The error message string <code>s</code> can later be
   * retrieved by the <code>{@link java.lang.Throwable#getMessage}</code>
   * method of class <code>java.lang.Throwable</code>.
   *
   * @param s the detail message.
   */
  public FilePermissionException(String s) {
    super(s);
  }
}
