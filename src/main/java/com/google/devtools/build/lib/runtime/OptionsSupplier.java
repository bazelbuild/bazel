// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.common.options.OptionsBase;

/** A supplier of command-line options. */
public interface OptionsSupplier {

  /**
   * Returns the startup options this module contributes.
   *
   * <p>This method will be called extremely early during server startup.
   */
  public Iterable<Class<? extends OptionsBase>> getStartupOptions();

  /**
   * Returns extra options this module contributes to a specific command. Note that option
   * inheritance applies: if this method returns a non-empty list, then the returned options are
   * added to every command that depends on this command.
   *
   * <p>This method may be called at any time, and the returned value may be cached. Implementations
   * must be thread-safe and never return different lists for the same command name. Typical
   * implementations look like this:
   *
   * <pre>
   * return commandName.equals("build")
   *     ? ImmutableList.of(MyOptions.class)
   *     : ImmutableList.of();
   * </pre>
   *
   * <p>Note that this example adds options to all commands that inherit from the build command.
   *
   * <p>This method is also used to generate command-line documentation; in order to avoid
   * duplicated options descriptions, this method should never return the same options class for two
   * different commands if one of them inherits the other.
   *
   * <p>If you want to add options to all commands, override {@link #getCommonCommandOptions}
   * instead.
   *
   * @param commandName the command name, e.g. "build" or "test".
   */
  Iterable<Class<? extends OptionsBase>> getCommandOptions(String commandName);

  /** Returns extra options this module contributes to all commands. */
  Iterable<Class<? extends OptionsBase>> getCommonCommandOptions();
}
