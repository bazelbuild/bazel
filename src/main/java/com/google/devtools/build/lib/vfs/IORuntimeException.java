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
// All Rights Reserved.

package com.google.devtools.build.lib.vfs;

import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Signals that an I/O exception of some sort has occurred. Contrary to
 * <code>java.io.IOException</code>, this class is a subclass of
 * <code>RuntimeException</code>, which allows you to signal an I/O problem
 * without polluting the callers. For details on why checked exceptions is bad,
 * try searching for "java checked exception mistake" on Google.
 */
public class IORuntimeException extends RuntimeException {
  /**
   * Constructs a new IORuntimeException with null as its detail message.
   */
  public IORuntimeException() {
    super();
  }

  /**
   * Constructs a new IORuntimeException with the specified detail message.
   */
  public IORuntimeException(String message) {
    super(message);
  }

  /**
   * Constructs a new IORuntimeException with the specified detail message and
   * cause.
   *
   * @param message the detail message, which is saved for later retrieval by
   *        the <code>Throwable.getMessage()</code> method.
   * @param cause the cause (which is saved for later retrieval by the
   *        <code>Throwable.getCause()</code> method). (A null value is
   *        permitted, and indicates that the cause is nonexistent or unknown.)
   */
  public IORuntimeException(String message, Throwable cause) {
    super(message, cause);
  }

  /**
   * Constructs a new IORuntimeException as a wrapper on a root cause
   */
  public IORuntimeException(Throwable cause) {
    super(cause);
  }

  /**
   * Returns the actual IOException that caused this exception, or null if it was not caused by an
   * IOException. Call <code>getCause()</code> instead if it was caused by other types of
   * exceptions.
   */
  @Nullable
  public IOException getCauseIOException() {
    Throwable cause = getCause();
    if (cause instanceof IOException ioException) {
      return ioException;
    } else {
      return null;
    }
  }

  private static final long serialVersionUID = 1L;
}
