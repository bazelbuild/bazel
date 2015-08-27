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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.LoggingUtil;

import java.util.logging.Level;

/**
 * Exceptions thrown during evaluation of BUILD ASTs or Skylark extensions.
 *
 * <p>This exception must always correspond to a repeatable, permanent error, i.e. evaluating the
 * same package again must yield the same exception. Notably, do not use this for reporting I/O
 * errors.
 *
 * <p>This requirement is in place so that we can cache packages where an error is reported by way
 * of {@link EvalException}.
 */
public class EvalException extends Exception {

  private Location location;
  private final String message;
  private final boolean dueToIncompleteAST;

  /**
   * @param location the location where evaluation/execution failed.
   * @param message the error message.
   */
  public EvalException(Location location, String message) {
    this.location = location;
    this.message = Preconditions.checkNotNull(message);
    this.dueToIncompleteAST = false;
  }

  /**
   * @param location the location where evaluation/execution failed.
   * @param message the error message.
   * @param dueToIncompleteAST if the error is caused by a previous error, such as parsing.
   */
  public EvalException(Location location, String message, boolean dueToIncompleteAST) {
    this.location = location;
    this.message = Preconditions.checkNotNull(message);
    this.dueToIncompleteAST = dueToIncompleteAST;
  }

  /**
   * @param location the location where evaluation/execution failed.
   * @param message the error message.
   * @param cause a Throwable that caused this exception.
   */
  public EvalException(Location location, String message, Throwable cause) {
    super(cause);
    this.location = location;
    // This is only used from Skylark, it's useful for debugging. Note that this only happens
    // when the Precondition below kills the execution anyway.
    if (message == null) {
      message = "";
    }
    if (cause != null) {
      String causeMsg = cause.getMessage();
      if (causeMsg != null && !message.contains(causeMsg)) {
        message = message + (message.isEmpty() ? "" : ": ") + causeMsg;
      }
    }
    if (message.isEmpty()) {
      LoggingUtil.logToRemote(Level.SEVERE, "Invalid EvalException", cause);
      throw new IllegalArgumentException("Invalid EvalException");
    }
    this.message = message;
    this.dueToIncompleteAST = false;
  }

  public EvalException(Location location, Throwable cause) {
    this(location, null, cause);
  }

  /**
   * Returns the error message with location info if exists.
   */
  public String print() { // TODO(bazel-team): do we also need a toString() method?
    return (getLocation() == null ? "" : getLocation()) + ": "
        + (message == null ? "" : message + "\n")
        + (dueToIncompleteAST ? "due to incomplete AST\n" : "")
        + getCauseMessage();
  }
  
  private String getCauseMessage() {
    Throwable cause = getCause();
    if (cause == null) {
      return "";
    }
    String causeMessage = cause.getMessage();
    return (causeMessage == null || message.contains(causeMessage)) ? "" : causeMessage;
  }

  /**
   * Returns the error message.
   */
  @Override
  public String getMessage() {
    return message;
  }

  /**
   * Returns the location of the evaluation error.
   */
  public Location getLocation() {
    return location;
  }

  /**
   * Returns a boolean that tells whether this exception was due to an incomplete AST
   */
  public boolean isDueToIncompleteAST() {
    return dueToIncompleteAST;
  }

  /**
   * Ensures that this EvalException has proper location information.
   * Does nothing if the exception already had a location, or if no location is provided.
   * @return this EvalException, in fluent style.
   */
  public EvalException ensureLocation(Location loc) {
    if (location == null && loc != null) {
      location = loc;
    }
    return this;
  }

  /**
   * A class to support a special case of EvalException when the cause of the error is an
   * Exception during a direct Java call. Allow the throwing code to provide context in a message.
   */
  public static final class EvalExceptionWithJavaCause extends EvalException {

    /**
     * @param location the location where evaluation/execution failed.
     * @param message the error message.
     * @param cause a Throwable that caused this exception.
     */
    public EvalExceptionWithJavaCause(Location location, String message, Throwable cause) {
      super(location, message, cause);
    }

    /**
     * @param location the location where evaluation/execution failed.
     * @param cause a Throwable that caused this exception.
     */
    public EvalExceptionWithJavaCause(Location location, Throwable cause) {
      this(location, null, cause);
    }
  }
}
