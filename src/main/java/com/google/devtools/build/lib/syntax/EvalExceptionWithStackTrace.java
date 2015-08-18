// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Rule;

/**
 * EvalException with a stack trace
 */
public class EvalExceptionWithStackTrace extends EvalException {
  private final StringBuilder builder = new StringBuilder();
  private Location location;
  private boolean printStackTrace;

  public EvalExceptionWithStackTrace(Exception original, Location callLocation) {
    super(callLocation, original.getMessage(), original.getCause());
    setLocation(callLocation);
    builder.append(super.getMessage());
    printStackTrace = false;
  }

  /**
   * Adds a line for the given function to the stack trace. Requires that #setLocation() was called
   * previously.
   */
  public void registerFunction(BaseFunction function) {
    addStackFrame(function.getFullName());
  }

  /**
   * Adds a line for the given rule to the stack trace.
   */
  public void registerRule(Rule rule) {
    setLocation(rule.getLocation());
    addStackFrame(String.format("%s(name = '%s', ...)", rule.getRuleClass(), rule.getName()));
  }

  /**
   * Adds a line for the given scope (function or rule).
   */
  private void addStackFrame(String scope) {
    builder.append(String.format("\n\tin %s [%s]", scope, location));
    printStackTrace |= (location != Location.BUILTIN);
  }

  /**
   * Sets the location for the next function to be added via #registerFunction().
   */
  public void setLocation(Location callLocation) {
    this.location = callLocation;
  }

  /**
   * Returns the exception message without the stack trace.
   */
  public String getOriginalMessage() {
    return super.getMessage();
  }

  @Override
  public String getMessage() {
    return print();
  }

  @Override
  public String print() {
    // Only print the stack trace when it contains more than one built-in function.
    return printStackTrace ? builder.toString() : getOriginalMessage();
  }
}
