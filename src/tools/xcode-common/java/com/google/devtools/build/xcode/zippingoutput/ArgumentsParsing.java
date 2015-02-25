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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.xcode.util.Value;

import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.util.Locale;

/**
 * Arguments passed to the do-something-then-zip wrapper tool.
 */
class ArgumentsParsing extends Value<ArgumentsParsing> {

  private final Optional<String> error;
  private final Optional<Arguments> arguments;

  private ArgumentsParsing(Optional<String> error, Optional<Arguments> arguments) {
    super(error, arguments);
    this.error = error;
    this.arguments = arguments;
  }

  public Optional<String> error() {
    return error;
  }

  public Optional<Arguments> arguments() {
    return arguments;
  }

  private static final int MIN_ARGS = 3;

  /**
   * @param args raw arguments passed to wrapper tool through the {@code main} method
   * @param subtoolName name of the subtool, such as "actool"
   * @return an instance based on results of parsing the given arguments.
   */
  public static ArgumentsParsing parse(FileSystem fileSystem, String[] args, String wrapperName,
      String subtoolName) {
    String capitalizedSubtool = subtoolName.toUpperCase(Locale.US);
    if (args.length < MIN_ARGS) {
      return new ArgumentsParsing(Optional.of(String.format(
              "Expected at least %1$d args.\n"
                  + "Usage: java %2$s OUTZIP ARCHIVEROOT (%3$s_CMD %3$s ARGS)\n"
                  + "Runs %4$s and zips the results.\n"
                  + "OUTZIP - the path to place the output zip file.\n"
                  + "ARCHIVEROOT - the path in the zip to place the output, or an empty\n"
                  + "    string for the root of the zip. e.g. 'Payload/foo.app'. If\n"
                  + "    this tool outputs a single file, ARCHIVEROOT is the name of\n"
                  + "    the only file in the zip file.\n"
                  + "%3$s_CMD - path to the subtool.\n"
                  + "    e.g. /Applications/Xcode.app/Contents/Developer/usr/bin/actool\n"
                  + "%3$s ARGS - the arguments to pass to %4$s besides the\n"
                  + "    one that specifies the output directory.\n",
              MIN_ARGS, wrapperName, capitalizedSubtool, subtoolName)),
          Optional.<Arguments>absent());
    }
    String outputZip = args[0];
    String archiveRoot = args[1];
    String subtoolCmd = args[2];
    if (archiveRoot.startsWith("/")) {
      return new ArgumentsParsing(
          Optional.of(String.format("Archive root cannot start with /: '%s'\n", archiveRoot)),
          Optional.<Arguments>absent());
    }

    // TODO(bazel-team): Remove this hack when the released version of Bazel uses the correct momc
    // path for device builds.
    subtoolCmd = subtoolCmd.replace("/iPhoneOS.platform/", "/iPhoneSimulator.platform/");
    if (!Files.isRegularFile(fileSystem.getPath(subtoolCmd))) {
      return new ArgumentsParsing(
          Optional.of(String.format(
              "The given %s_CMD does not exist: '%s'\n", capitalizedSubtool, subtoolCmd)),
          Optional.<Arguments>absent());
    }
    return new ArgumentsParsing(
        Optional.<String>absent(),
        Optional.<Arguments>of(
            new Arguments(
                outputZip, archiveRoot, subtoolCmd,
                ImmutableList.copyOf(args).subList(MIN_ARGS, args.length))));
  }
}
