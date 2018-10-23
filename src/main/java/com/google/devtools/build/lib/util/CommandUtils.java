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

import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

/**
 * Utility methods relating to the {@link Command} class.
 */
public class CommandUtils {

  private CommandUtils() {} // Prevent instantiation.

  private static Collection<String> commandLine(Command command) {
    return Arrays.asList(command.getCommandLineElements());
  }

  private static Map<String, String> env(Command command) {
    return command.getEnvironmentVariables();
  }

  private static String cwd(Command command) {
    return command.getWorkingDirectory() == null ? null : command.getWorkingDirectory().getPath();
  }

  /**
   * Construct an error message that describes a failed command invocation.
   * Currently this returns a message of the form "error executing command foo
   * bar baz".
   */
  public static String describeCommandError(boolean verbose, Command command) {
    return CommandFailureUtils.describeCommandError(verbose, commandLine(command), env(command),
                                                    cwd(command));
  }

  /**
   * Construct an error message that describes a failed command invocation.
   * Currently this returns a message of the form "foo failed: error executing
   * command /dir/foo bar baz".
   */
  public static String describeCommandFailure(boolean verbose, Command command) {
    return CommandFailureUtils.describeCommandFailure(verbose, commandLine(command), env(command),
                                                      cwd(command));
  }

  /**
   * Construct an error message that describes a failed command invocation.
   * Currently this returns a message of the form "foo failed: error executing
   * command /dir/foo bar baz: exception message", with the
   * command's stdout and stderr output appended if available.
   */
  public static String describeCommandFailure(boolean verbose, CommandException exception) {
    String message = describeCommandFailure(verbose, exception.getCommand()) + ": "
        + exception.getMessage();
    if (exception instanceof AbnormalTerminationException) {
      CommandResult result = ((AbnormalTerminationException) exception).getResult();
      try {
        return message + "\n"
            + new String(result.getStdout())
            + new String(result.getStderr());
      } catch (IllegalStateException e) {
        // This can happen if the command didn't save stdout/stderr,
        // so ignore this exception and fall through to the ordinary case.
      }
    }
    return message;
  }

}
