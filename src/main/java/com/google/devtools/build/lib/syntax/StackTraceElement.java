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
import com.google.devtools.build.lib.syntax.Argument.Passed;

import java.util.List;
import java.util.Map;

/**
 * Represents an element of {@link Environment}'s stack trace.
 */
// TODO(fwe): maybe combine this with EvalExceptionWithStackTrace.StackTraceElement
public final class StackTraceElement {
  private final Location location;
  private final String name;
  private final String nameArg;
  private final BaseFunction func;

  public StackTraceElement(BaseFunction func, Map<String, Object> kwargs) {
    this(func.getName(), func.getLocation(), func, getNameArg(kwargs));
  }

  public StackTraceElement(Identifier identifier, List<Passed> args) {
    this(identifier.getName(), identifier.getLocation(), null, getNameArg(args));
  }

  private StackTraceElement(String name, Location location, BaseFunction func, String nameArg) {
    this.name = name;
    this.location = location;
    this.func = func;
    this.nameArg = nameArg;
  }

  /**
   * Returns the value of the argument 'name' (or null if there is none).
   */
  private static String getNameArg(Map<String, Object> kwargs) {
    Object value = (kwargs == null) ? null : kwargs.get("name");
    return (value == null) ? null : value.toString();
  }

  /**
   * Returns the value of the argument 'name' (or null if there is none).
   */
  private static String getNameArg(List<Passed> args) {
    for (Argument.Passed arg : args) {
      if (arg != null) {
        String name = arg.getName();
        if (name != null && name.equals("name")) {
          Expression expr = arg.getValue();
          return (expr == null) ? null : expr.toString();
        }
      }
    }
    return null;
  }

  public String getName() {
    return name;
  }

  public Location getLocation() {
    return location;
  }

  /**
   * Returns a more expressive description of this element, if possible.
   */
  public String getLabel() {
    return (nameArg == null) ? getName() : String.format("%s(name = %s)", name, nameArg);
  }

  public boolean hasFunction(BaseFunction func) {
    return this.func != null && this.func.equals(func);
  }

  @Override
  public String toString() {
    return String.format(
        "%s @ %s", getLabel(), (location == null) ? "<unknown>" : location.toString());
  }
}
