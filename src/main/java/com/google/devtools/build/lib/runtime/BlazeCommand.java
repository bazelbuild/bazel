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

import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;

/**
 * Interface implemented by Blaze commands. In addition to implementing this interface, each command
 * must be annotated with a {@link Command} annotation.
 */
public interface BlazeCommand {
  /**
   * This method provides the imperative portion of the command. It takes a {@link
   * OptionsParsingResult} instance {@code options}, which provides access to the options instances
   * via {@link OptionsParsingResult#getOptions(Class)}, and access to the residue (the remainder of
   * the command line) via {@link OptionsParsingResult#getResidue()}. The framework parses and makes
   * available exactly the options that the command class specifies via the annotation {@link
   * Command#options()}. The command indicates success / failure via its return value, which becomes
   * the Unix exit status of the Blaze client process. It may indicate that the server needs to be
   * shut down or that a particular binary needs to be exec()ed on the terminal where Blaze was
   * invoked from by returning the appropriate {@link BlazeCommandResult}.
   *
   * @param env The environment for the current command invocation
   * @param options A parsed options instance initialized with the values for the options specified
   *     in {@link Command#options()}.
   * @return The Unix exit status for the Blaze client.
   */
  BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options);

  /**
   * Allows the command to provide command-specific option defaults and/or requirements. This method
   * is called after all command-line and rc file options have been parsed.
   *
   * @param optionsParser the options parser for the current command
   */
  default void editOptions(OptionsParser optionsParser) {}
}
