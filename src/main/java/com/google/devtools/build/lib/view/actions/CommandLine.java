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

package com.google.devtools.build.lib.view.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.CollectionUtils;

/**
 * A representation of a command line to be executed by a SpawnAction.
 */
public abstract class CommandLine {
  /**
   * Returns the command line.
   */
  public abstract Iterable<String> arguments();

  /**
   * Returns whether the command line represents a shell command with the
   * given shell executable. This is used to give better error messages.
   *
   * <p>By default, this method returns false.
   */
  public boolean isShellCommand() {
    return false;
  }

  /**
   * A default implementation of a command line backed by a copy of the given
   * list of arguments.
   */
  static CommandLine ofInternal(Iterable<String> arguments,
                                final boolean isShellCommand) {
    final Iterable<String> immutableArguments = CollectionUtils.makeImmutable(arguments);
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() {
        return immutableArguments;
      }

      @Override
      public boolean isShellCommand() {
        return isShellCommand;
      }
    };
  }

  /**
   * A default implementation of a command line backed by a copy of the given
   * list of arguments.
   */
  public static CommandLine of(Iterable<String> arguments, final boolean isShellCommand) {
    final Iterable<String> immutableArguments = CollectionUtils.makeImmutable(arguments);
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() {
        return immutableArguments;
      }

      @Override
      public boolean isShellCommand() {
        return isShellCommand;
      }
    };
  }

  /**
   * Returns a mixed implementation using the list of arguments followed by
   * the command line. The {@link CommandLine#isShellCommand} is implemented
   * by only looking at the arguments list.
   */
  static CommandLine ofMixed(final ImmutableList<String> executableArgs,
      final CommandLine commandLine, final boolean isShellCommand) {
    Preconditions.checkState(!executableArgs.isEmpty());
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() {
        return Iterables.concat(executableArgs, commandLine.arguments());
      }

      @Override
      public boolean isShellCommand() {
        return isShellCommand;
      }
    };
  }
}
