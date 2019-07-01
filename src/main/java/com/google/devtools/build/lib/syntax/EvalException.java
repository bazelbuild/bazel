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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.LoggingUtil;
import java.util.logging.Level;
import javax.annotation.Nullable;

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

  @Nullable private Location location;
  private final String message;
  private final boolean dueToIncompleteAST;

  private static final Joiner LINE_JOINER = Joiner.on("\n").skipNulls();
  private static final Joiner FIELD_JOINER = Joiner.on(": ").skipNulls();

  /**
   * @param location the location where evaluation/execution failed.
   * @param message the error message.
   */
  public EvalException(Location location, String message) {
    this.location = location;
    this.message = Preconditions.checkNotNull(message);
    this.dueToIncompleteAST = false;
  }

  public EvalException(Location location, String message, String url) {
    this.location = location;
    this.dueToIncompleteAST = false;
    this.message =
        Preconditions.checkNotNull(message)
            + "\n"
            + "Need help? See "
            + Preconditions.checkNotNull(url);
  }

  /**
   * @param location the location where evaluation/execution failed.
   * @param message the error message.
   * @param dueToIncompleteAST if the error is caused by a previous error, such as parsing.
   */
  public EvalException(Location location, String message, boolean dueToIncompleteAST) {
    this(location, message, dueToIncompleteAST, true);
  }

  /**
   * Create an EvalException with the option to not fill in the java stack trace. An optimization
   * for ReturnException, and potentially others, which aren't exceptional enough to include a
   * stack trace.
   *
   * @param location the location where evaluation/execution failed.
   * @param message the error message.
   * @param dueToIncompleteAST if the error is caused by a previous error, such as parsing.
   * @param fillInJavaStackTrace whether or not to fill in the java stack trace for this exception
   */
  EvalException(
      Location location,
      String message,
      boolean dueToIncompleteAST,
      boolean fillInJavaStackTrace) {
    super(null, null, /*enableSuppression=*/ true, fillInJavaStackTrace);
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
    // This is only used from Skylark, it's useful for debugging.
    this.message = FIELD_JOINER.join(message, getCauseMessage(message));
    if (this.message.isEmpty()) {
      String details;
      if (cause == null) {
        details = "Invalid EvalException: no cause given!";
      } else {
        details = "Invalid EvalException:\n" + Throwables.getStackTraceAsString(cause);
      }
      LoggingUtil.logToRemote(Level.SEVERE, details, cause);
      throw new IllegalArgumentException(details);
    }
    this.dueToIncompleteAST = false;
  }

  public EvalException(Location location, Throwable cause) {
    this(location, null, cause);
  }

  /**
   * Returns the error message with location info if exists.
   */
  public String print() { // TODO(bazel-team): do we also need a toString() method?
    return LINE_JOINER.join("\n", FIELD_JOINER.join(getLocation(), message),
        (dueToIncompleteAST ? "due to incomplete AST" : ""),
        getCauseMessage(message));
  }

  /**
   * @param message the message of this exception, so far.
   * @return a message for the cause of the exception, if the main message (passed as argument)
   * doesn't already contain this cause; return null if no new information is available.
   */
  private String getCauseMessage(String message) {
    Throwable cause = getCause();
    if (cause == null) {
      return null;
    }
    String causeMessage = cause.getMessage();
    if (causeMessage == null) {
      return null;
    }
    if (message == null) {
      return causeMessage;
    }
    // Skip the cause if it is redundant with the message so far.
    if (message.contains(causeMessage)) {
      return null;
    }
    return causeMessage;
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
  @Nullable
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
   * Returns whether this exception can be added to a stack trace created by {@link
   * EvalExceptionWithStackTrace}.
   */
  public boolean canBeAddedToStackTrace() {
    return true;
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
