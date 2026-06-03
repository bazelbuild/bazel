// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.execgraph;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.github.luben.zstd.ZstdInputStream;
import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.ExecutionGraph.Node;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import com.google.devtools.build.lib.util.io.MessageInputStreamWrapper.BinaryInputStreamWrapper;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
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

/**
 * A tool to inspect and parse the Bazel execution graph log (written by
 * {@code --experimental_enable_execution_graph_log}).
 *
 * <p>The log is a zstd-compressed stream of length-delimited {@code execution_graph.Node} protos.
 */
public final class ExecGraphParser {
  private ExecGraphParser() {}

  private static final String DELIMITER =
      "\n---------------------------------------------------------\n";

  @VisibleForTesting
  static MessageInputStream<Node> getMessageInputStream(String path) throws IOException {
    // The execution graph log is always a zstd-compressed stream of length-delimited Node protos.
    return new BinaryInputStreamWrapper<>(
        new ZstdInputStream(new FileInputStream(path)), Node.getDefaultInstance());
  }

  @VisibleForTesting
  static class FilteredStream implements MessageInputStream<Node> {
    final MessageInputStream<Node> in;
    final String restrictToRunner;

    FilteredStream(MessageInputStream<Node> in, String restrictToRunner) {
      this.in = in;
      this.restrictToRunner = restrictToRunner;
    }

    @Override
    @Nullable
    public Node read() throws IOException {
      Node node;
      while ((node = in.read()) != null) {
        if (restrictToRunner == null || restrictToRunner.equals(node.getRunner())) {
          return node;
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
  static String getKey(Node node) {
    // Nodes don't list their outputs, so the human-readable description is the most stable key for
    // matching the same node across two builds.
    String description = node.getDescription();
    return description.isEmpty() ? null : description;
  }

  @VisibleForTesting
  static class OrderedStream implements MessageInputStream<Node> {

    public static class Golden {
      // A map of positions of nodes in the first file.
      // Key: the node's description.
      // Value: the position of the node in the file (e.g., 0th, 1st etc).
      private final Map<String, Integer> positions;
      private int index;

      public Golden() {
        positions = new HashMap<>();
        index = 0;
      }

      public void addNode(Node node) {
        String key = getKey(node);
        if (key != null) {
          positions.put(key, index++);
        }
      }

      public int positionFor(Node node) {
        String key = getKey(node);
        if (key != null && positions.containsKey(key)) {
          return positions.get(key);
        }
        return -1;
      }
    }

    private static class Element {
      public int position;
      public Node element;

      public Element(int position, Node element) {
        this.position = position;
        this.element = element;
      }
    }

    private final MessageInputStream<Node> in;
    private final Golden golden;

    OrderedStream(Golden golden, MessageInputStream<Node> in) throws IOException {
      this.in = in;
      this.golden = golden;
      processInputFile();
    }

    // nodes from input that appear in golden, indexed by their position in the golden.
    PriorityQueue<Element> sameNodes;
    // nodes in input that are not in the golden, in order received.
    Queue<Node> uniqueNodes;

    private void processInputFile() throws IOException {
      sameNodes = new PriorityQueue<>((e1, e2) -> (e1.position - e2.position));
      uniqueNodes = new ArrayDeque<>();

      Node node;
      while ((node = in.read()) != null) {
        int position = golden.positionFor(node);
        if (position >= 0) {
          sameNodes.add(new Element(position, node));
        } else {
          uniqueNodes.add(node);
        }
      }
    }

    @Override
    public Node read() {
      if (sameNodes.isEmpty()) {
        return uniqueNodes.poll();
      }
      return sameNodes.remove().element;
    }

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  public static void output(
      MessageInputStream<Node> in, OutputStream outStream, OrderedStream.Golden golden)
      throws IOException {
    PrintWriter out =
        new PrintWriter(new BufferedWriter(new OutputStreamWriter(outStream, UTF_8)), true);
    Node node;
    while ((node = in.read()) != null) {
      out.println(node);
      out.println(DELIMITER);
      if (golden != null) {
        golden.addNode(node);
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

    try (MessageInputStream<Node> input = getMessageInputStream(logPath)) {
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
      try (MessageInputStream<Node> file2 = getMessageInputStream(secondPath);
          OutputStream output = new FileOutputStream(output2)) {
        MessageInputStream<Node> parser = new FilteredStream(file2, options.restrictToRunner);
        // OrderedStream will read the whole golden on initialization,
        // so it is safe to close after.
        parser = new OrderedStream(golden, parser);
        output(parser, output, null);
      }
    }
  }
}
