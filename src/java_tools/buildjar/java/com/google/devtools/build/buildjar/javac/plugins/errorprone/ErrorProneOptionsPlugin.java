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

package com.google.devtools.build.buildjar.javac.plugins.errorprone;

import com.google.devtools.build.buildjar.InvalidCommandLineException;
import com.google.devtools.build.buildjar.javac.plugins.BlazeJavaCompilerPlugin;
import com.google.errorprone.ErrorProneOptions;
import com.google.errorprone.InvalidCommandLineOptionException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Process (and discard) Error Prone specific options.
 * 
 * <p>This is a stop-gap until full Error Prone support is added to Bazel.
 */
public class ErrorProneOptionsPlugin extends BlazeJavaCompilerPlugin {

  @Override
  public List<String> processArgs(List<String> args) throws InvalidCommandLineException {
    // TODO(cushon): add -XepIgnoreUnknownCheckNames once Error Prone is supported
    return processEpOptions(processExtraChecksOption(args));
  }

  private List<String> processEpOptions(List<String> args) throws InvalidCommandLineException {
    ErrorProneOptions epOptions;
    try {
      epOptions = ErrorProneOptions.processArgs(args);
    } catch (InvalidCommandLineOptionException e) {
      throw new InvalidCommandLineException(e.getMessage());
    }
    return Arrays.asList(epOptions.getRemainingArgs());
  }

  private List<String> processExtraChecksOption(List<String> args) {
    List<String> arguments = new ArrayList<>();
    for (String arg : args) {
      switch (arg) {
        case "-extra_checks":
        case "-extra_checks:on":
          break;
        case "-extra_checks:off":
          break;
        default:
          arguments.add(arg);
      }
    }
    return arguments;
  }
}
