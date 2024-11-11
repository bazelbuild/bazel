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
import com.google.devtools.build.lib.exec.SpawnLogReconstructor;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import com.google.devtools.build.lib.util.io.MessageInputStreamWrapper.BinaryInputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageInputStreamWrapper.JsonInputStreamWrapper;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import javax.annotation.Nullable;

/** A tool to inspect and parse the Bazel execution log. */
public final class ExecLogParser {
  private ExecLogParser() {}

  private static final String DELIMITER =
      "\n---------------------------------------------------------\n";

  private static byte[] readFirstFourBytes(String path) throws IOException {
    try (InputStream in = new FileInputStream(path)) {
      return in.readNBytes(4);
    }
  }

  @VisibleForTesting
  static MessageInputStream<SpawnExec> getMessageInputStream(String path) throws IOException {
    byte[] b = readFirstFourBytes(path);
    if (b.length == 4
        && b[0] == 0x28
        && b[1] == (byte) 0xb5
        && b[2] == 0x2f
        && b[3] == (byte) 0xfd) {
      // Looks like a compact file (zstd-compressed).
      // This is definitely not a JSON file (the first byte is not '{') and definitely not a
      // binary file (the first byte would indicate the size of the first message, and the
      // second byte would indicate an invalid wire type).
      return new SpawnLogReconstructor(new FileInputStream(path));
    }
    if (b.length >= 2 && b[0] == '{' && b[1] == '\n') {
      // Looks like a JSON file.
      // This is definitely not a compact file (the first byte is not 0x28) and definitely not a
      // binary file (the first byte would indicate the size of the first message, and the
      // second byte would indicate a field with number 1 and wire type I32, which doesn't match
      // the proto definition).
      return new JsonInputStreamWrapper<>(
          new FileInputStream(path), SpawnExec.getDefaultInstance());
    }
    // Otherwise assume it's a binary file.
    return new BinaryInputStreamWrapper<>(
        new FileInputStream(path), SpawnExec.getDefaultInstance());
  }

  @VisibleForTesting
  static class FilteredStream implements MessageInputStream<SpawnExec> {
    final MessageInputStream<SpawnExec> in;
    final String restrictToRunner;

    FilteredStream(MessageInputStream<SpawnExec> in, String restrictToRunner) {
      this.in = in;
      this.restrictToRunner = restrictToRunner;
    }

    @Override
    @Nullable
    public SpawnExec read() throws IOException {
      SpawnExec ex;
      while ((ex = in.read()) != null) {
        if (restrictToRunner == null || restrictToRunner.equals(ex.getRunner())) {
          return ex;
        }
      }
      return null;
    }

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  @Nullable
  static String getFirstOutput(SpawnExec e) {
    if (e.getListedOutputsCount() > 0) {
      return e.getListedOutputs(0);
    }
    return null;
  }

  @VisibleForTesting
  static class OrderedStream implements MessageInputStream<SpawnExec> {

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

    private final MessageInputStream<SpawnExec> in;
    private final Golden golden;

    OrderedStream(Golden golden, MessageInputStream<SpawnExec> in) throws IOException {
      this.in = in;
      this.golden = golden;
      processInputFile();
    }

    // actions from input that appear in golden, indexed by their position in the golden.
    PriorityQueue<Element> sameActions;
    // actions in input that are not in the golden, in order received.
    Queue<SpawnExec> uniqueActions;

    private void processInputFile() throws IOException {
      sameActions = new PriorityQueue<>((e1, e2) -> (e1.position - e2.position));
      uniqueActions = new ArrayDeque<>();

      SpawnExec ex;
      while ((ex = in.read()) != null) {
        int position = golden.positionFor(ex);
        if (position >= 0) {
          sameActions.add(new Element(position, ex));
        } else {
          uniqueActions.add(ex);
        }
      }
    }

    @Override
    public SpawnExec read() {
      if (sameActions.isEmpty()) {
        return uniqueActions.poll();
      }
      return sameActions.remove().element;
    }

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  public static void output(
      MessageInputStream<SpawnExec> in, OutputStream outStream, OrderedStream.Golden golden)
      throws IOException {
    PrintWriter out =
        new PrintWriter(new BufferedWriter(new OutputStreamWriter(outStream, UTF_8)), true);
    SpawnExec ex;
    while ((ex = in.read()) != null) {
      out.println(ex);
      out.println(DELIMITER);
      if (golden != null) {
        golden.addSpawnExec(ex);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    OptionsParser op = OptionsParser.builder().optionsClasses(ParserOptions.class).build();
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

    OrderedStream.Golden golden = null;
    if (secondPath != null) {
      golden = new OrderedStream.Golden();
    }

    try (MessageInputStream<SpawnExec> input = getMessageInputStream(logPath)) {
      FilteredStream parser = new FilteredStream(input, options.restrictToRunner);

      if (output1 == null) {
        output(parser, System.out, golden);
      } else {
        try (OutputStream output = new FileOutputStream(output1)) {
          output(parser, output, golden);
        }
      }
    }

    if (secondPath != null) {
      try (MessageInputStream<SpawnExec> file2 = getMessageInputStream(secondPath);
          OutputStream output = new FileOutputStream(output2)) {
        MessageInputStream<SpawnExec> parser = new FilteredStream(file2, options.restrictToRunner);
        // ReorderingParser will read the whole golden on initialization,
        // so it is safe to close after.
        parser = new OrderedStream(golden, parser);
        output(parser, output, null);
      }
    }
  }
}

