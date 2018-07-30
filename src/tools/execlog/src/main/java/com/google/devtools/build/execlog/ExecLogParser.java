// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.execlog;

import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.common.options.OptionsParser;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * A tool to inspect and parse the Bazel execution log.
 */
final class ExecLogParser {

  static final String DELIMITER = "\n---------------------------------------------------------\n";

  final InputStream in;
  final String restrictToRunner;

  ExecLogParser(InputStream in, String restrictToRunner) {
    this.in = in;
    this.restrictToRunner = restrictToRunner;
  }

  public SpawnExec getNext() throws IOException {
    SpawnExec ex;
    // Find the next record whose runner matches
    do {
      if (in.available() <= 0) {
        // End of file
        return null;
      }
      ex = SpawnExec.parseDelimitedFrom(in);
    } while (restrictToRunner != null && !restrictToRunner.equals(ex.getRunner()));
    return ex;
  }

  public static void main(String[] args) throws Exception {
    OptionsParser op = OptionsParser.newOptionsParser(ParserOptions.class);
    op.parseAndExitUponError(args);

    ParserOptions options = op.getOptions(ParserOptions.class);
    if (options.logPath == null) {
      System.err.println("--log_path needs to be specified.");
      System.exit(1);
    }

    ExecLogParser parser =
        new ExecLogParser(new FileInputStream(options.logPath), options.restrictToRunner);

    SpawnExec ex;
    while ((ex = parser.getNext()) != null) {
      System.out.println(ex);
      System.out.println(DELIMITER);
    }
  }
}

