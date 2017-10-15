// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.benchmark;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;
import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Main class for running benchmark. */
public class Main {

  private static final Logger logger = Logger.getLogger(Main.class.getName());

  public static void main(String[] args) {

    BenchmarkOptions opt = null;
    try {
      opt = parseArgs(args);
    } catch (Exception e) {
      if (!e.getMessage().isEmpty()) {
        logger.log(Level.SEVERE, e.getMessage());
      }
      System.exit(1);
    }

    // Prepare paths
    File workspace = new File(opt.workspace);
    if (workspace.isFile()) {
      logger.log(Level.SEVERE, "Workspace directory is an existing file: " + opt.workspace);
      System.exit(1);
    }
    if (!workspace.exists() && !workspace.mkdirs()) {
      logger.log(Level.SEVERE, "Failed to create workspace directory: " + opt.workspace);
      System.exit(1);
    }
    File outputFile = new File(opt.output);
    if (outputFile.exists()) {
      logger.log(Level.SEVERE, "Output file already exists: " + opt.output);
      System.exit(1);
    }

    BuildGroupRunner runner = new BuildGroupRunner(workspace.toPath());
    BuildGroupResult result = null;
    try {
      result = runner.run(opt);
    } catch (Exception e) {
      logger.log(Level.SEVERE, e.getMessage());
      System.exit(1);
    }

    // Store data
    try {
      Writer writer = Files.newBufferedWriter(outputFile.toPath(), UTF_8);
      JsonFormat.printer().appendTo(result, writer);
      writer.flush();
    } catch (InvalidProtocolBufferException e) {
      logger.log(Level.SEVERE, "Invalid protobuf: " + e.getMessage());
      System.exit(1);
    } catch (IOException e) {
      logger.log(Level.SEVERE, "Failed to write to output file: " + e.getMessage());
      System.exit(1);
    }
  }

  public static BenchmarkOptions parseArgs(String[] args) throws OptionsParsingException {
    BenchmarkOptions opt = Options.parse(BenchmarkOptions.class, args).getOptions();

    // Missing options
    if (opt.workspace.isEmpty() || opt.output.isEmpty()) {
      System.err.println(Options.getUsage(BenchmarkOptions.class));
      throw new IllegalArgumentException("Argument --workspace and --output should not be empty.");
    }
    // Should use exact one argument between from/to, after/before and versions
    int emptyNum = booleanToInt(opt.versionFilter == null)
        + booleanToInt(opt.dateFilter == null)
        + booleanToInt(opt.versions.isEmpty());
    if (emptyNum != 2) {
      System.err.println(Options.getUsage(BenchmarkOptions.class));
      throw new IllegalArgumentException("Please use exact one type of version filter at a time.");
    }

    return opt;
  }

  private static int booleanToInt(boolean b) {
    return b ? 1 : 0;
  }
}
