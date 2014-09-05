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

package com.google.devtools.build.xcode.zippingoutput;

/**
 * Defines a zipping wrapper for a tool. A wrapper is a tool that runs some subcommand that writes
 * its output (usually multiple files in a non-trivial directory structure) to some directory, and
 * then zips the output of that subtool into a timestamp-less zip file whose location is specified
 * on the command line.
 * <p>
 * The arguments passed are passed in this form:
 * <pre>
 * java {wrapper_name} OUTZIP ARCHIVEROOT (SUBTOOL_CMD SUBTOOL ARGS)
 * </pre>
 * Where SUBTOOL ARGS are the arguments to pass to SUBTOOL_CMD unchanged, except the argument that
 * specifies the output directory. Note that the arguments are positional and do not use flags in
 * the form of -f or --foo, except for those flags passed directly to and interpreted by the
 * subtool itself.
 * <p>
 * A wrapper has some simple metadata the name of the wrapped tool
 * and how to transform {@link Arguments} passed to the wrapper tool into arguments for the subtool.
 */
public interface Wrapper {
  /** The name of the wrapper, such as {@code ActoolZip}. */
  String name();

  /** The subtool name, such as {@code actool}. */
  String subtoolName();

  /**
   * Returns the command (i.e. argv), including the executable file, to be executed.
   * @param arguments the parsed arguments passed to the wrapper tool
   * @param outputDirectory the output directory which the subtool should write tool
   */
  Iterable<String> subCommand(Arguments arguments, String outputDirectory);

  /**
   * Indicates whether the output directory must exist before invoking the wrapped tool, this being
   * dependent on the nature of the tool only. Note that the directory immediately containing the
   * output directory is guaranteed to always exist, regardless of this method's return value.
   */
  boolean outputDirectoryMustExist();
}
