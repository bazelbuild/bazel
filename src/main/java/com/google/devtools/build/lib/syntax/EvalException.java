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

  private final Location location;
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

  private EvalException(Location location, Throwable cause) {
    super(cause);
    this.location = location;
    // This is only used from Skylark, it's useful for debugging. Note that this only happens
    // when the Precondition below kills the execution anyway.
    if (cause.getMessage() == null) {
      cause.printStackTrace();
    }
    this.message = Preconditions.checkNotNull(cause.getMessage());
    this.dueToIncompleteAST = false;
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

  public boolean isDueToIncompleteAST() {
    return dueToIncompleteAST;
  }

  /**
   * A class to support a special case of EvalException when the cause of the error is an
   * Exception during a direct Java call.
   */
  public static final class EvalExceptionWithJavaCause extends EvalException {

    public EvalExceptionWithJavaCause(Location location, Throwable cause) {
      super(location, cause);
    }
  }

  /**
   * Returns the error message with location info if exists.
   */
  public String print() {
    return getLocation() == null ? getMessage() : getLocation().print() + ": " + getMessage();
  }
}
