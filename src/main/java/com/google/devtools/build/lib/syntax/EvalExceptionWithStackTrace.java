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
// This requires several steps:
// - break dependency on syntax. Use only call frames (function names and locations).
//   That means to print the source we must reopen the file, which may require
//   heuristics as it may reside in a fake file system.
// - eliminate or at least de-emphasize EvalException constructors that accept
//   a Location. Most exceptions should be created with no location; the function
//   call machinery fills in the details the first time exception bubbles up out of
//   a call. The only exceptions (ha!) are (a) exceptions thrown by the interpreter at a
//   particular statement/operator location, and (b) when client code wants to add
//   a fake alternative (or additional?) location, such as the place a data structure
//   was created.
// - get rid of the "phantom node" machinery. In the rare cases where it is needed,
//   it is trivial to call a built-in wrapper function that presents the desired
//   name and location and simply calls the desired function. There is no need to
//   complicate the API.
// - clarify the various string methods (toString, getMessage, getOriginalMessage, print),
//   and don't let this subclass totally redefine them.
//   Printing the stack should be an explicit operation, as in Java.
// - make this class private; the parent should define the complete API.
//   (For internal catch statements, it may be helpful to keep the two classes.)
public class EvalExceptionWithStackTrace extends EvalException {

  private StackFrame mostRecentElement;

  // Called only from Eval.maybeTransformException.
  EvalExceptionWithStackTrace(EvalException original, Node culprit) {
    // The 'message' here must be non-empty in case getCause() is null,
    // as the super(-fragile) constructor crashes if both are empty.
    super(nodeLocation(culprit), getNonEmptyMessage(original), original.getCause());
    registerNode(culprit);
  }

  @Override
  protected boolean canBeAddedToStackTrace() {
    // Doesn't make any sense to add this exception to another instance of
    // EvalExceptionWithStackTrace.
    return false;
  }

  private static Location nodeLocation(Node node) {
    return node instanceof CallExpression
        ? ((CallExpression) node).getLparenLocation()
        : node.getStartLocation();
  }

  /** Adds an entry for the given {@code Node} to the stack trace. */
  void registerNode(Node node) {
    addStackFrame(node.toString().trim(), nodeLocation(node), /*canPrint=*/ true);
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
    addStackFrame(function.getName(), function.getLocation(), /*canPrint=*/ true);
    addStackFrame(callDescription, location, /*canPrint=*/ false);
  }

  /** Adds a line for the given frame. */
  private void addStackFrame(String text, Location location, boolean canPrint) {
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
        && location.file().equals(mostRecentElement.location.file())
        && location.line() == mostRecentElement.location.line()) {
      return;
    }
    mostRecentElement = new StackFrame(text, location, mostRecentElement, canPrint);
  }

  /**
   * Returns the exception message without the stack trace.
   */
  public String getOriginalMessage() {
    return super.getMessage();
  }

  @Override
  public String getMessage() {
    // TODO(adonovan): don't change the meaning of getMessage (and toString) in the subclass.
    // Printing the stack should be an explicit operation.
    return print();
  }

  @Override
  public String print() {
    // Currently, we do not limit the text length per line.

    // Prints the stack trace iff it contains more than just one built-in function.
    return canPrintStackTrace()
        ? StackTracePrinter.print(getOriginalMessage(), mostRecentElement)
        : getOriginalMessage();
  }

  /**
   * Returns true when there is at least one non-built-in element.
   */
  protected boolean canPrintStackTrace() {
    return mostRecentElement != null && mostRecentElement.cause != null;
  }

  /**
   * An element in the stack trace which contains the name of the offending function / rule /
   * statement and its location.
   */
  private static final class StackFrame {
    final String text;
    final Location location;
    final StackFrame cause; // tail of linked list
    final boolean canPrint;

    StackFrame(String text, Location location, StackFrame cause, boolean canPrint) {
      this.text = text;
      this.location = location;
      this.cause = cause;
      this.canPrint = canPrint;
    }

    @Override
    public String toString() {
      return String.format("%s @ %s -> %s", text, location, String.valueOf(cause));
    }
  }

  /** A collection of stateless of functions to print stack traces similar to Python. */
  private static final class StackTracePrinter {

    /** Turns the given message and StackTraceElements into a string. */
    static String print(String message, StackFrame mostRecentElement) {
      Deque<String> output = new LinkedList<>();

      // Adds dummy element for the rule call that uses the location of the top-most function.
      mostRecentElement =
          new StackFrame(
              "",
              mostRecentElement.location,
              mostRecentElement.cause == null ? null : mostRecentElement,
              true);

      while (mostRecentElement != null) {
        if (mostRecentElement.canPrint) {
          String entry = printElement(mostRecentElement);
          if (entry != null && entry.length() > 0) {
            addEntry(output, entry);
          }
        }

        mostRecentElement = mostRecentElement.cause;
      }

      addMessage(output, message);
      return Joiner.on(System.lineSeparator()).join(output);
    }

    /** Returns the string representation of the given element. */
    static String printElement(StackFrame element) {
      // Similar to Python, the first (most-recent) entry in the stack frame is printed only once.
      // Consequently, we skip it here.
      if (element.cause == null) {
        return "";
      }

      // Prints a two-line string, similar to Python.
      Location location = getLocation(element.cause);
      return String.format(
          "\tFile \"%s\", line %d%s%n\t\t%s",
          printPath(location), getLine(location), printFunction(element.text), element.cause.text);
    }

    /** Returns the location of the given element or Location.BUILTIN if the element is null. */
    static Location getLocation(StackFrame element) {
      return element == null ? Location.BUILTIN : element.location;
    }

    static String printFunction(String func) {
      if (func.isEmpty()) {
        return "";
      }

      int pos = func.indexOf('(');
      return String.format(", in %s", (pos < 0) ? func : func.substring(0, pos));
    }

    static String printPath(Location loc) {
      return loc == null ? "<unknown>" : loc.file();
    }

    static int getLine(Location loc) {
      return loc == null ? 0 : loc.line();
    }

    /** Adds the given string to the specified Deque. */
    static void addEntry(Deque<String> output, String toAdd) {
      output.addLast(toAdd);
    }

    /**
     * Adds the given message to the given output dequeue after all stack trace elements have been
     * added.
     */
    static void addMessage(Deque<String> output, String message) {
      output.addFirst("Traceback (most recent call last):");
      output.addLast(message);
    }
  }

  /**
   * Returns a non-empty message for the given exception.
   *
   * <p>If the exception itself does not have a message, a new message is constructed from the
   * exception's class name. For example, an IllegalArgumentException will lead to "Illegal
   * Argument". Additionally, the location in the Java code will be added, if applicable,
   */
  // TODO(adonovan): eliminate this function. We have no business interpreting Java exceptions.
  private static String getNonEmptyMessage(EvalException original) {
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
