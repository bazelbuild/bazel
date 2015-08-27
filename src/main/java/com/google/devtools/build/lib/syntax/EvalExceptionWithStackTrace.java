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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Deque;
import java.util.LinkedList;

/**
 * EvalException with a stack trace
 */
public class EvalExceptionWithStackTrace extends EvalException {

  private StackTraceElement mostRecentElement;

  public EvalExceptionWithStackTrace(Exception original, Location callLocation) {
    super(callLocation, getNonEmptyMessage(original), original.getCause());
  }

  /**
   * Adds an entry for the given statement to the stack trace.
   */
  public void registerStatement(Statement statement) {
    Preconditions.checkState(
        mostRecentElement == null, "Cannot add a statement to a non-empty stack trace.");
    addStackFrame(statement.toString().trim(), statement.getLocation());
  }

  /**
   * Adds an entry for the given function to the stack trace.
   */
  public void registerFunction(BaseFunction function, Location location) {
    addStackFrame(function.getFullName(), location);
  }

  /**
   * Adds an entry for the given rule to the stack trace.
   */
  public void registerRule(Rule rule) {
    addStackFrame(
        String.format("%s(name = '%s')", rule.getRuleClass(), rule.getName()), rule.getLocation());
  }

  /**
   * Adds a line for the given frame.
   */
  private void addStackFrame(String label, Location location) {
    mostRecentElement = new StackTraceElement(label, location, mostRecentElement);
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
    return print(StackTracePrinter.INSTANCE);
  }

  /**
   *  Prints the stack trace iff it contains more than just one built-in function.
   */
  public String print(StackTracePrinter printer) {
    return canPrintStackTrace()
        ? printer.print(getOriginalMessage(), mostRecentElement)
        : getOriginalMessage();
  }

  /**
   * Returns true when there is at least one non-built-in element.
   */
  protected boolean canPrintStackTrace() {
    return mostRecentElement != null && mostRecentElement.getCause() != null;
  }

  /**
   * An element in the stack trace which contains the name of the offending function / rule /
   * statement and its location.
   */
  protected final class StackTraceElement {
    private final String label;
    private final Location location;
    private final StackTraceElement cause;

    StackTraceElement(String label, Location location, StackTraceElement cause) {
      this.label = label;
      this.location = location;
      this.cause = cause;
    }

    String getLabel() {
      return label;
    }

    Location getLocation() {
      return location;
    }

    StackTraceElement getCause() {
      return cause;
    }
  }

  /**
   * Singleton class that prints stack traces similar to Python.
   */
  public enum StackTracePrinter {
    INSTANCE;

    /**
     * Turns the given message and StackTraceElements into a string.
     */
    public final String print(String message, StackTraceElement mostRecentElement) {
      Deque<String> output = new LinkedList<>();

      while (mostRecentElement != null) {
        String entry = print(mostRecentElement);
        if (entry != null && entry.length() > 0) {
          addEntry(output, entry);
        }

        mostRecentElement = mostRecentElement.getCause();
      }

      addMessage(output, message);
      return Joiner.on("\n").join(output);
    }

    /**
     * Returns the location which should be shown on the same line as the label of the given
     * element.
     */
    protected Location getDisplayLocation(StackTraceElement element) {
      // If there is a rule definition in this element, it should print its own location in
      // the BUILD file instead of using a location in a bzl file.
      return describesRule(element) ? element.getLocation() : getLocation(element.getCause());
    }

    /**
     * Returns the location of the given element or Location.BUILTIN if the element is null.
     */
    private Location getLocation(StackTraceElement element) {
      return (element == null) ? Location.BUILTIN : element.getLocation();
    }

    /**
     * Returns whether the given element describes the rule definition in a BUILD file.
     */
    protected boolean describesRule(StackTraceElement element) {
      PathFragment pathFragment = element.getLocation().getPath();
      return pathFragment != null && pathFragment.getPathString().contains("BUILD");
    }

    /**
     * Returns the string representation of the given element.
     */
    protected String print(StackTraceElement element) {
      // Similar to Python, the first (most-recent) entry in the stack frame is printed only once.
      // Consequently, we skip it here.
      if (element.getCause() == null) {
        return "";
      }

      // Prints a two-line string, similar to Python.
      Location location = getDisplayLocation(element);
      return String.format(
          "\tFile \"%s\", line %d, in %s%n\t\t%s",
          printPath(location.getPath()),
          location.getStartLineAndColumn().getLine(),
          element.getLabel(),
          element.getCause().getLabel());
    }

    private String printPath(PathFragment path) {
      return (path == null) ? "<unknown>" : path.getPathString();
    }

    /**
     * Adds the given string to the specified Deque.
     */
    protected void addEntry(Deque<String> output, String toAdd) {
      output.addLast(toAdd);
    }

    /**
     * Adds the given message to the given output dequeue after all stack trace elements have been
     * added.
     */
    protected void addMessage(Deque<String> output, String message) {
      output.addFirst("Traceback (most recent call last):");
      output.addLast(message);
    }
  }

  /**
   * Returns a non-empty message for the given exception.
   *
   * <p> If the exception itself does not have a message, a new message is constructed from the
   * exception's class name.
   * For example, an IllegalArgumentException will lead to "Illegal Argument".
   */
  private static String getNonEmptyMessage(Exception original) {
    Preconditions.checkNotNull(original);
    String msg = original.getMessage();
    if (msg != null && !msg.isEmpty()) {
      return msg;
    }

    char[] name = original.getClass().getSimpleName().replace("Exception", "").toCharArray();
    boolean first = true;
    StringBuilder builder = new StringBuilder();

    for (char current : name) {
      if (Character.isUpperCase(current) && !first) {
        builder.append(" ");
      }
      builder.append(current);
      first = false;
    }

    return builder.toString();
  }
}
