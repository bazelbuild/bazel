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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
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
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * A tool to inspect and parse the Bazel execution log.
 */
final class ExecLogParser {

  static final String DELIMITER = "\n---------------------------------------------------------\n";

  @VisibleForTesting
  interface Parser {
    SpawnExec getNext() throws IOException;
  }

  @VisibleForTesting
  static class FilteringLogParser implements Parser {
    final InputStream in;
    final String restrictToRunner;

    FilteringLogParser(InputStream in, String restrictToRunner) {
      this.in = in;
      this.restrictToRunner = restrictToRunner;
    }

    @Override
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
  }

  static String getFirstOutput(SpawnExec e) {
    if (e.getListedOutputsCount() > 0) {
      return e.getListedOutputs(0);
    }
    return null;
  }

  @VisibleForTesting
  static class ReorderingParser implements Parser {

    public static class Golden {
      // A map of positions of actions in the first file.
      // Key: first output filename of the action
      // Value: the position of the action in the file (e.g., 0th, 1st etc).
      private final Map<String, Integer> positions;
      private int index;

      public Golden() {
        positions = new HashMap<>();
        index = 0;
      }

      public void addSpawnExec(SpawnExec ex) {
        String key = getFirstOutput(ex);
        if (key != null) {
          positions.put(key, index++);
        }
      }

      public int positionFor(SpawnExec ex) {
        String key = getFirstOutput(ex);
        if (key != null && positions.containsKey(key)) {
          return positions.get(key);
        }
        return -1;
      }
    }

    private static class Element {
      public int position;
      public SpawnExec element;

      public Element(int position, SpawnExec element) {
        this.position = position;
        this.element = element;
      }
    }

    private final Golden golden;

    ReorderingParser(Golden golden, Parser input) throws IOException {
      this.golden = golden;
      processInputFile(input);
    }

    // actions from input that appear in golden, indexed by their position in the golden.
    PriorityQueue<Element> sameActions;
    // actions in input that are not in the golden, in order received.
    Queue<SpawnExec> uniqueActions;

    private void processInputFile(Parser input) throws IOException {
      sameActions = new PriorityQueue<>((e1, e2) -> (e1.position - e2.position));
      uniqueActions = new ArrayDeque<>();

      SpawnExec ex;
      while ((ex = input.getNext()) != null) {
        int position = golden.positionFor(ex);
        if (position >= 0) {
          sameActions.add(new Element(position, ex));
        } else {
          uniqueActions.add(ex);
        }
      }
    }

    @Override
    public SpawnExec getNext() {
      if (sameActions.isEmpty()) {
        return uniqueActions.poll();
      }
      return sameActions.remove().element;
    }
  }

  public static void output(Parser p, OutputStream outStream, ReorderingParser.Golden golden)
      throws IOException {
    PrintWriter out =
        new PrintWriter(new BufferedWriter(new OutputStreamWriter(outStream, UTF_8)), true);
    SpawnExec ex;
    while ((ex = p.getNext()) != null) {
      out.println(ex);
      out.println(DELIMITER);
      if (golden != null) {
        golden.addSpawnExec(ex);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    OptionsParser op = OptionsParser.newOptionsParser(ParserOptions.class);
    op.parseAndExitUponError(args);

    ParserOptions options = op.getOptions(ParserOptions.class);
    List<String> remainingArgs = op.getResidue();

    if (!remainingArgs.isEmpty()) {
      System.err.println("Unexpected options: " + String.join(" ", remainingArgs));
      System.exit(1);
    }

    if (options.logPath == null || options.logPath.isEmpty()) {
      System.err.println("--log_path needs to be specified.");
      System.exit(1);
    }
    if (options.outputPath != null && options.outputPath.size() > options.logPath.size()) {
      System.err.println("Too many --output_path values.");
      System.exit(1);
    }

    String logPath = options.logPath.get(0);
    String secondPath = null;
    String output1 = null;
    String output2 = null;

    if (options.logPath.size() > 1) {
      if (options.logPath.size() > 2) {
        System.err.println("Too many --log_path: at most two files are currently supported.");
        System.exit(1);
      }
      secondPath = options.logPath.get(1);
      if (options.outputPath == null || options.outputPath.size() != 2) {
        System.err.println(
            "Exactly two --output_path values expected, one for each of --log_path values.");
        System.exit(1);
      }
      output1 = options.outputPath.get(0);
      output2 = options.outputPath.get(1);
    } else {
      if (options.outputPath != null && !options.outputPath.isEmpty()) {
        output1 = options.outputPath.get(0);
      }
    }

    ReorderingParser.Golden golden = null;
    if (secondPath != null) {
      golden = new ReorderingParser.Golden();
    }

    try (InputStream input = Files.newInputStream(Paths.get(logPath))) {
      Parser parser = new FilteringLogParser(input, options.restrictToRunner);

      if (output1 == null) {
        output(parser, System.out, golden);
      } else {
        try (OutputStream output = Files.newOutputStream(Paths.get(output1))) {
          output(parser, output, golden);
        }
      }
    }

    if (secondPath != null) {
      try (InputStream file2 = Files.newInputStream(Paths.get(secondPath));
          OutputStream output = Files.newOutputStream(Paths.get(output2))) {
        Parser parser = new FilteringLogParser(file2, options.restrictToRunner);
        // ReorderingParser will read the whole golden on initialization,
        // so it is safe to close after.
        parser = new ReorderingParser(golden, parser);
        output(parser, output, null);
      }
    }
  }
}

