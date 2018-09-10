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
package com.google.devtools.build.workspacelog;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos.WorkspaceEvent;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** A tool to inspect and parse the Bazel workspace rules log. */
final class WorkspaceLogParser {

  static final String DELIMITER = "\n---------------------------------------------------------\n";

  @VisibleForTesting
  static class ExcludingLogParser {
    final InputStream in;
    final Set<String> excludedRules;

    ExcludingLogParser(InputStream in, Set<String> excludedRules) {
      this.in = in;
      if (excludedRules == null) {
        this.excludedRules = Collections.emptySet();
      } else {
        this.excludedRules = excludedRules;
      }
    }

    public WorkspaceEvent getNext() throws IOException {
      WorkspaceEvent w;
      // Find the next record whose runner matches
      do {
        if (in.available() <= 0) {
          // End of file
          return null;
        }
        w = WorkspaceEvent.parseDelimitedFrom(in);
      } while (excludedRules.contains(w.getRule()));
      return w;
    }
  }

  public static void output(ExcludingLogParser p, OutputStream outStream) throws IOException {
    PrintWriter out =
        new PrintWriter(new BufferedWriter(new OutputStreamWriter(outStream, UTF_8)), true);
    WorkspaceEvent w;
    while ((w = p.getNext()) != null) {
      out.println(w);
      out.println(DELIMITER);
    }
  }

  public static void main(String[] args) throws Exception {
    OptionsParser op = OptionsParser.newOptionsParser(WorkspaceLogParserOptions.class);
    op.parseAndExitUponError(args);

    WorkspaceLogParserOptions options = op.getOptions(WorkspaceLogParserOptions.class);
    List<String> remainingArgs = op.getResidue();

    if (!remainingArgs.isEmpty()) {
      System.err.println("Unexpected options: " + String.join(" ", remainingArgs));
      System.exit(1);
    }

    if (options.logPath == null || options.logPath.isEmpty()) {
      System.err.println("--log_path needs to be specified.");
      System.exit(1);
    }

    try (InputStream input = Files.newInputStream(Paths.get(options.logPath))) {
      ExcludingLogParser parser;
      if (options.excludeRule == null) {
        parser = new ExcludingLogParser(input, null);
      } else {
        parser = new ExcludingLogParser(input, new HashSet<String>(options.excludeRule));
      }

      if (options.outputPath == null) {
        output(parser, System.out);
      } else {
        try (OutputStream output = Files.newOutputStream(Paths.get(options.outputPath))) {
          output(parser, output);
        }
      }
    }
  }
}
