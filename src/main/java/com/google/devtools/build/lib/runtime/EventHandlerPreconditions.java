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

package com.google.devtools.build.lib.runtime;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.ExceptionListener;
import com.google.devtools.build.lib.util.LoggingUtil;

import java.util.logging.Level;

/**
 * Reports precondition failures from within an event handler.
 * Necessary because the EventBus silently ignores exceptions thrown from within a handler.
 * This class logs the exceptions and creates some noise when a precondition check fails.
 */
public class EventHandlerPreconditions {

  private final ExceptionListener listener;

  /**
   * Creates a new precondition helper which outputs errors to the given reporter.
   */
  public EventHandlerPreconditions(ExceptionListener listener) {
    this.listener = listener;
  }

  /**
   * Verifies that the given condition (a check on an argument) is true,
   * throwing an IllegalArgumentException if not.
   *
   * @param condition a condition to check for truth.
   * @throws IllegalArgumentException if the condition is false.
   */
  @SuppressWarnings("unused")
  public void checkArgument(boolean condition) {
    checkArgument(condition, null);
  }

  /**
   * Verifies that the given condition (a check on an argument) is true,
   * throwing an IllegalArgumentException with the given message if not.
   *
   * @param condition a condition to check for truth.
   * @param message extra information to output if the condition is false.
   * @throws IllegalArgumentException if the condition is false.
   */
  public void checkArgument(boolean condition, String message) {
    try {
      Preconditions.checkArgument(condition, message);
    } catch (IllegalArgumentException iae) {
      String error = "Event handler argument check failed";
      LoggingUtil.logToRemote(Level.SEVERE, error, iae);
      listener.error(null, error, iae);
      throw iae; // Still terminate the handler.
    }
  }

  /**
   * Verifies that the given condition (a check against the program's current state) is true,
   * throwing an IllegalStateException if not.
   *
   * @param condition a condition to check for truth.
   * @throws IllegalStateException if the condition is false.
   */
  public void checkState(boolean condition) {
    checkState(condition, null);
  }

  /**
   * Verifies that the given condition (a check against the program's current state) is true,
   * throwing an IllegalStateException with the given message if not.
   *
   * @param condition a condition to check for truth.
   * @param message extra information to output if the condition is false.
   * @throws IllegalStateException if the condition is false.
   */
  public void checkState(boolean condition, String message) {
    try {
      Preconditions.checkState(condition, message);
    } catch (IllegalStateException ise) {
      String error = "Event handler state check failed";
      LoggingUtil.logToRemote(Level.SEVERE, error, ise);
      listener.error(null, error, ise);
      throw ise; // Still terminate the handler.
    }
  }

  /**
   * Fails with an IllegalStateException when invoked.
   */
  public void fail(String message) {
    String error = "Event handler failed: " + message;
    IllegalStateException ise = new IllegalStateException(message);
    LoggingUtil.logToRemote(Level.SEVERE, error, ise);
    listener.error(null, error, ise);
    throw ise;
  }

  /**
   * Verifies that the given argument is not null, throwing a NullPointerException if it is null.
   * Returns the original argument or throws.
   *
   * @param object an object to test for null.
   * @return the reference which was checked.
   * @throws NullPointerException if the object is null.
   */
  public <T> T checkNotNull(T object) {
    return checkNotNull(object, null);
  }

  /**
   * Verifies that the given argument is not null, throwing a
   * NullPointerException with the given message if it is null.
   * Returns the original argument or throws.
   *
   * @param object an object to test for null.
   * @param message extra information to output if the object is null.
   * @return the reference which was checked.
   * @throws NullPointerException if the object is null.
   */
  public <T> T checkNotNull(T object, String message) {
    try {
      return Preconditions.checkNotNull(object, message);
    } catch (NullPointerException npe) {
      String error = "Event handler not-null check failed";
      LoggingUtil.logToRemote(Level.SEVERE, error, npe);
      listener.error(null, error, npe);
      throw npe;
    }
  }
}
