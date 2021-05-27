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

package com.google.devtools.build.lib.util;

import static java.util.Map.Entry.comparingByKey;

import com.google.common.base.Preconditions;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import java.io.File;
import java.util.Collection;
import java.util.Comparator;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Utility methods for describing command failures.
 * See also the CommandUtils class.
 * Unlike that one, this class does not depend on Command;
 * instead, it just manipulates command lines represented as
 * Collection&lt;String&gt;.
 */
public class CommandFailureUtils {

  // Interface that provides building blocks when describing command.
  private interface DescribeCommandImpl {
    void describeCommandBeginIsolate(StringBuilder message);
    void describeCommandEndIsolate(StringBuilder message);
    void describeCommandCwd(String cwd, StringBuilder message);
    void describeCommandEnvPrefix(StringBuilder message, boolean isolated);
    void describeCommandEnvVar(StringBuilder message, Map.Entry<String, String> entry);
    /**
     * Formats the command element and adds it to the message.
     *
     * @param message the message to modify
     * @param commandElement the command element to be added to the message
     * @param isBinary is true if the `commandElement` is the binary to be executed
     */
    void describeCommandElement(StringBuilder message, String commandElement, boolean isBinary);

    void describeCommandExec(StringBuilder message);
  }

  private static final class LinuxDescribeCommandImpl implements DescribeCommandImpl {

    @Override
    public void describeCommandBeginIsolate(StringBuilder message) {
      message.append("(");
    }

    @Override
    public void describeCommandEndIsolate(StringBuilder message) {
      message.append(")");
    }

    @Override
    public void describeCommandCwd(String cwd, StringBuilder message) {
      message.append("cd ").append(ShellEscaper.escapeString(cwd)).append(" && \\\n  ");
    }

    @Override
    public void describeCommandEnvPrefix(StringBuilder message, boolean isolated) {
      message.append(isolated
          ? "env - \\\n  "
          : "env \\\n  ");
    }

    @Override
    public void describeCommandEnvVar(StringBuilder message, Map.Entry<String, String> entry) {
      message.append(ShellEscaper.escapeString(entry.getKey())).append('=')
          .append(ShellEscaper.escapeString(entry.getValue())).append(" \\\n  ");
    }

    @Override
    public void describeCommandElement(
        StringBuilder message, String commandElement, boolean isBinary) {
      message.append(ShellEscaper.escapeString(commandElement));
    }

    @Override
    public void describeCommandExec(StringBuilder message) {
      message.append("exec ");
    }
  }

  // TODO(bazel-team): (2010) Add proper escaping. We can't use ShellUtils.shellEscape() as it is
  // incompatible with CMD.EXE syntax, but something else might be needed.
  private static final class WindowsDescribeCommandImpl implements DescribeCommandImpl {

    @Override
    public void describeCommandBeginIsolate(StringBuilder message) {
      // TODO(bazel-team): Implement this.
    }

    @Override
    public void describeCommandEndIsolate(StringBuilder message) {
      // TODO(bazel-team): Implement this.
    }

    @Override
    public void describeCommandCwd(String cwd, StringBuilder message) {
      message.append("cd ").append("/d ").append(cwd).append("\n");
    }

    @Override
    public void describeCommandEnvPrefix(StringBuilder message, boolean isolated) { }

    @Override
    public void describeCommandEnvVar(StringBuilder message, Map.Entry<String, String> entry) {
      message.append("SET ").append(entry.getKey()).append('=')
          .append(entry.getValue()).append("\n  ");
    }

    @Override
    public void describeCommandElement(
        StringBuilder message, String commandElement, boolean isBinary) {
      // Replace the forward slashes with back slashes if the `commandElement` is the binary path
      message.append(isBinary ? commandElement.replace('/', '\\') : commandElement);
    }

    @Override
    public void describeCommandExec(StringBuilder message) {
      // TODO(bazel-team): Implement this if possible for greater efficiency.
    }
  }

  private static final DescribeCommandImpl describeCommandImpl =
      OS.getCurrent() == OS.WINDOWS ? new WindowsDescribeCommandImpl()
                                    : new LinuxDescribeCommandImpl();
  private static final int APPROXIMATE_MAXIMUM_MESSAGE_LENGTH = 200;

  private CommandFailureUtils() {} // Prevent instantiation.

  /**
   * Construct a string that describes the command.
   * Currently this returns a message of the form "foo bar baz",
   * with shell meta-characters appropriately quoted and/or escaped,
   * prefixed (if verbose is true) with an "env" command to set the environment.
   *
   * @param form Form of the command to generate; see the documentation of the
   * {@link CommandDescriptionForm} values.
   */
  public static String describeCommand(
      CommandDescriptionForm form,
      boolean prettyPrintArgs,
      Collection<String> commandLineElements,
      @Nullable Map<String, String> environment,
      @Nullable String cwd) {

    Preconditions.checkNotNull(form);
    StringBuilder message = new StringBuilder();
    int size = commandLineElements.size();
    int numberRemaining = size;

    if (form == CommandDescriptionForm.COMPLETE) {
      describeCommandImpl.describeCommandBeginIsolate(message);
    }

    if (form != CommandDescriptionForm.ABBREVIATED) {
      if (cwd != null) {
        describeCommandImpl.describeCommandCwd(cwd, message);
      }
      /*
       * On Linux, insert an "exec" keyword to save a fork in "blaze run"
       * generated scripts.  If we use "env" as a wrapper, the "exec" needs to
       * be applied to the entire "env" invocation.
       *
       * On Windows, this is a no-op.
       */
      describeCommandImpl.describeCommandExec(message);
      /*
       * Java does not provide any way to invoke a subprocess with the environment variables
       * in a specified order.  The order of environment variables in the 'environ' array
       * (which is set by the 'envp' parameter to the execve() system call)
       * is determined by the order of iteration on a HashMap constructed inside Java's
       * ProcessBuilder class (in the ProcessEnvironment class), which is nondeterministic.
       *
       * Nevertheless, we *print* the environment variables here in sorted order, rather
       * than in the potentially nondeterministic order that will be actually used.
       * This is slightly dubious... in theory a process's behaviour could depend on the order
       * of the environment variables passed to it.  (For example, the order of environment
       * variables in the environ array affects the output of '/usr/bin/env'.)
       * However, in practice very few processes depend on the order of the environment
       * variables, and using a deterministic sorted order here makes Blaze's output more
       * deterministic and easier to read.  So this seems the lesser of two evils... I think.
       * Anyway, it's not like we have much choice... even if we wanted to, there's no way to
       * print out the nondeterministic order that will actually be used, since there's
       * no way to guarantee that the iteration over entrySet() here will return the same
       * sequence as the iteration over entrySet() inside the ProcessBuilder class
       * (in ProcessEnvironment.StringEnvironment.toEnvironmentBlock()).
       */
      if (environment != null) {
        describeCommandImpl.describeCommandEnvPrefix(
            message, form != CommandDescriptionForm.COMPLETE_UNISOLATED);
        // A map can never have two keys with the same value, so we only need to compare the keys.
        Comparator<Map.Entry<String, String>> mapEntryComparator = comparingByKey();
        for (Map.Entry<String, String> entry :
            Ordering.from(mapEntryComparator).sortedCopy(environment.entrySet())) {
          message.append("  ");
          describeCommandImpl.describeCommandEnvVar(message, entry);
        }
      }
    }

    boolean isFirstArgument = true;
    for (String commandElement : commandLineElements) {
      if (form == CommandDescriptionForm.ABBREVIATED
          && message.length() + commandElement.length() > APPROXIMATE_MAXIMUM_MESSAGE_LENGTH) {
        message
            .append(" ... (remaining ")
            .append(numberRemaining)
            .append(numberRemaining == 1 ? " argument" : " arguments")
            .append(" skipped)");
        break;
      } else {
        if (numberRemaining < size) {
          message.append(prettyPrintArgs ? " \\\n    " : " ");
        }
        describeCommandImpl.describeCommandElement(message, commandElement, isFirstArgument);
        numberRemaining--;
      }
      isFirstArgument = false;
    }

    if (form == CommandDescriptionForm.COMPLETE) {
      describeCommandImpl.describeCommandEndIsolate(message);
    }

    return message.toString();
  }

  /**
   * Construct an error message that describes a failed command invocation. Currently this returns a
   * message of the form "error executing command foo bar baz".
   */
  public static String describeCommandError(
      boolean verbose,
      Collection<String> commandLineElements,
      Map<String, String> env,
      String cwd,
      @Nullable PlatformInfo executionPlatform) {

    CommandDescriptionForm form = verbose
        ? CommandDescriptionForm.COMPLETE
        : CommandDescriptionForm.ABBREVIATED;

    StringBuilder output = new StringBuilder();
    output.append("error executing command ");
    if (verbose) {
      output.append("\n  ");
    }
    output.append(
        describeCommand(form, /* prettyPrintArgs= */ false, commandLineElements, env, cwd));
    if (verbose && executionPlatform != null) {
      output.append("\n");
      output.append("Execution platform: ").append(executionPlatform.label());
    }
    return output.toString();
  }

  /**
   * Construct an error message that describes a failed command invocation. Currently this returns a
   * message of the form "foo failed: error executing command /dir/foo bar baz".
   */
  public static String describeCommandFailure(
      boolean verbose,
      Collection<String> commandLineElements,
      Map<String, String> env,
      String cwd,
      @Nullable PlatformInfo executionPlatform) {

    String commandName = commandLineElements.iterator().next();
    // Extract the part of the command name after the last "/", if any.
    String shortCommandName = new File(commandName).getName();
    return shortCommandName
        + " failed: "
        + describeCommandError(verbose, commandLineElements, env, cwd, executionPlatform);
  }
}
