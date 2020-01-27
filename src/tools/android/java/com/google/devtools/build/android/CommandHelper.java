// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.io.CharStreams;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

/** A helper class for executing a command with {@link ProcessBuilder}. */
class CommandHelper {
  static String execute(String action, List<String> command) throws IOException {
    final StringBuilder processLog = new StringBuilder();

    final Process process = new ProcessBuilder().command(command).redirectErrorStream(true).start();
    processLog.append("Command: ");
    Joiner.on("\\\n\t").appendTo(processLog, command);
    processLog.append("\nOutput:\n");
    final InputStreamReader stdout = new InputStreamReader(process.getInputStream(), UTF_8);
    while (process.isAlive()) {
      processLog.append(CharStreams.toString(stdout));
    }
    // Make sure the full stdout is read.
    while (stdout.ready()) {
      processLog.append(CharStreams.toString(stdout));
    }
    if (process.exitValue() != 0) {
      throw new RuntimeException(String.format("Error during %s:", action) + "\n" + processLog);
    }
    return processLog.toString();
  }

  private CommandHelper() {}
}
