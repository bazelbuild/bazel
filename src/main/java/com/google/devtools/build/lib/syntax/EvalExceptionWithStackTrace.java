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
package com.google.devtools.build.lib.syntax;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import java.util.Deque;
import java.util.LinkedList;

/** EvalException with a stack trace. */
// TODO(adonovan): get rid of this. Every EvalException should record the stack.
public class EvalExceptionWithStackTrace extends EvalException {

  private StackFrame mostRecentElement;

  public EvalExceptionWithStackTrace(Exception original, Node culprit) {
    super(extractLocation(original, culprit), getNonEmptyMessage(original), getCause(original));
    registerNode(culprit);
  }

  @Override
  public boolean canBeAddedToStackTrace() {
    // Doesn't make any sense to add this exception to another instance of
    // EvalExceptionWithStackTrace.
    return false;
  }

  /**
   * Returns the appropriate location for this exception.
   *
   * <p>If the {@code Node} is non-null, its location is used. Otherwise, we try to get the location
   * of the exception.
   */
  private static Location extractLocation(Exception original, Node culprit) {
    if (culprit != null) {
      return nodeLocation(culprit);
    }
    return original instanceof EvalException ? ((EvalException) original).getLocation() : null;
  }

  private static Location nodeLocation(Node node) {
    return node instanceof CallExpression
        ? ((CallExpression) node).getLparenLocation()
        : node.getStartLocation();
  }

  /**
   * Returns the "real" cause of this exception.
   *
   * <p>If the original exception is an EvalException, its cause is returned.
   * Otherwise, the original exception itself is seen as the cause for this exception.
   */
  private static Throwable getCause(Exception ex) {
    return (ex instanceof EvalException) ? ex.getCause() : ex;
  }

  /** Adds an entry for the given {@code Node} to the stack trace. */
  void registerNode(Node node) {
    addStackFrame(node.toString().trim(), nodeLocation(node));
  }

  /**
   * Makes sure the stack trace is rooted in a function call.
   *
   * <p>In some cases (rule implementation application, aspect implementation application) bazel
   * calls into the function directly (using BaseFunction.call). In that case, since there is no
   * CallExpression to evaluate, stack trace mechanism cannot record this call. This method allows
   * to augument the stack trace with information about the call.
   */
  public void registerPhantomCall(
      String callDescription, Location location, StarlarkCallable function) {
    /*
     *
     * We add two new frames to the stack:
     * 1. Pseudo-function call (for example, rule definition)
     * 2. Function entry (Rule implementation)
     *
     * Similar to Python, all functions that were entered (except for the top-level ones) appear
     * twice in the stack trace output. This would lead to the following trace:
     *
     * File BUILD, line X, in <module>
     *     rule_definition()
     * File BUILD, line X, in rule_definition
     *     rule_implementation()
     * File bzl, line Y, in rule_implementation
     *     ...
     *
     * Please note that lines 3 and 4 are quite confusing since a) the transition from
     * rule_definition to rule_implementation happens internally and b) the locations do not make
     * any sense.
     * Consequently, we decided to omit lines 3 and 4 from the output via canPrint = false:
     *
     * File BUILD, line X, in <module>
     *     rule_definition()
     * File bzl, line Y, in rule_implementation
     *     ...
     *
     * */
    addStackFrame(function.getName(), function.getLocation());
    addStackFrame(callDescription, location, false);
  }

  /** Adds a line for the given frame. */
  private void addStackFrame(String label, Location location, boolean canPrint) {
    // TODO(bazel-team): This check was originally created to weed out duplicates in case the same
    // node is added twice, but it's not clear if that is still a possibility.
    //
    // [I suspect the real reason it was added is not because of duplicate nodes,
    // but because the stack corresponds to the stack of expressions in the tree-walking
    // evaluator's recursion, which often includes several subexpressions within
    // the same line, e.g. f().g()+1. If the stack had one entry per function call,
    // like StarlarkThread.CallStack, there would be no problem.
    // This was revealed when we started recording operator positions precisely,
    // causing the f(), .g(), and + operations in the example above to have different
    // locations within the same line. --adonovan]
    //
    // In any case, it would be better to eliminate the check and not create unwanted duplicates in
    // the first place.
    // The check is problematic because it suppresses tracebacks in the REPL,
    // where line numbers can be reset within a single session.
    if (mostRecentElement != null
        && location.file().equals(mostRecentElement.getLocation().file())
        && location.line() == mostRecentElement.getLocation().line()) {
      return;
    }
    mostRecentElement = new StackFrame(label, location, mostRecentElement, canPrint);
  }

  private void addStackFrame(String label, Location location)   {
    addStackFrame(label, location, true);
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
    // Currently, we do not limit the text length per line.
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
  protected static final class StackFrame {
    private final String label;
    private final Location location;
    private final StackFrame cause;
    private final boolean canPrint;

    StackFrame(String label, Location location, StackFrame cause, boolean canPrint) {
      this.label = label;
      this.location = location;
      this.cause = cause;
      this.canPrint = canPrint;
    }

    String getLabel() {
      return label;
    }

    Location getLocation() {
      return location;
    }

    StackFrame getCause() {
      return cause;
    }

    boolean canPrint() {
      return canPrint;
    }

    @Override
    public String toString() {
      return String.format("%s @ %s -> %s", label, location, String.valueOf(cause));
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
    public final String print(String message, StackFrame mostRecentElement) {
      Deque<String> output = new LinkedList<>();

      // Adds dummy element for the rule call that uses the location of the top-most function.
      mostRecentElement = new StackFrame("", mostRecentElement.getLocation(),
          (mostRecentElement.getCause() == null) ? null : mostRecentElement, true);

      while (mostRecentElement != null) {
        if (mostRecentElement.canPrint()) {
          String entry = print(mostRecentElement);
          if (entry != null && entry.length() > 0) {
            addEntry(output, entry);
          }
        }

        mostRecentElement = mostRecentElement.getCause();
      }

      addMessage(output, message);
      return Joiner.on(System.lineSeparator()).join(output);
    }

    /** Returns the string representation of the given element. */
    protected String print(StackFrame element) {
      // Similar to Python, the first (most-recent) entry in the stack frame is printed only once.
      // Consequently, we skip it here.
      if (element.getCause() == null) {
        return "";
      }

      // Prints a two-line string, similar to Python.
      Location location = getLocation(element.getCause());
      return String.format(
          "\tFile \"%s\", line %d%s%n\t\t%s",
          printPath(location),
          getLine(location),
          printFunction(element.getLabel()),
          element.getCause().getLabel());
    }

    /** Returns the location of the given element or Location.BUILTIN if the element is null. */
    private Location getLocation(StackFrame element) {
      return (element == null) ? Location.BUILTIN : element.getLocation();
    }

    private String printFunction(String func) {
      if (func.isEmpty()) {
        return "";
      }

      int pos = func.indexOf('(');
      return String.format(", in %s", (pos < 0) ? func : func.substring(0, pos));
    }

    private String printPath(Location loc) {
      return loc == null ? "<unknown>" : loc.file();
    }

    private int getLine(Location loc) {
      return loc == null ? 0 : loc.line();
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
   * Additionally, the location in the Java code will be added, if applicable,
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

    java.lang.StackTraceElement[] trace = original.getStackTrace();
    if (trace.length > 0) {
      builder.append(String.format(": %s.%s() in %s:%d", getShortClassName(trace[0]),
          trace[0].getMethodName(), trace[0].getFileName(), trace[0].getLineNumber()));
    }

    return builder.toString();
  }

  private static String getShortClassName(java.lang.StackTraceElement element) {
    String name = element.getClassName();
    int pos = name.lastIndexOf('.');
    return (pos < 0) ? name : name.substring(pos + 1);
  }
}
