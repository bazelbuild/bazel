// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.lcovmerger;

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Command line utility to convert raw coverage files to lcov (text) format.
 */
public class Main {
  private static final Logger logger = Logger.getLogger(Main.class.getName());

  public static void main(String[] args) {
    Map<String, String> flags = null;
    try {
      flags = parseFlags(args);
    } catch (IllegalArgumentException e) {
      logger.log(Level.SEVERE, e.getMessage());
      System.exit(1);
    }

    LcovMerger lcovMerger = new LcovMerger(flags.get("coverage_dir"), flags.get("output_file"));
    boolean success = lcovMerger.merge();
    System.exit(success ? 0 : 1);
  }

  /**
   * Parse flags in the form of "--coverage_dir=... -output_file=..."
   */
  private static Map<String, String> parseFlags(String[] args) {
    Map<String, String> flags = new HashMap<>();

    for (String arg : args) {
      if (!arg.startsWith("--")) {
        throw new IllegalArgumentException("Argument (" + arg + ") should start with --");
      }
      String[] parts = arg.substring(2).split("=", 2);
      if (parts.length != 2) {
        throw new IllegalArgumentException("There should be = in argument (" + arg + ")");
      }
      flags.put(parts[0], parts[1]);
    }

    // Validate flags
    if (!flags.containsKey("coverage_dir")) {
      throw new IllegalArgumentException("coverage_dir was not specified");
    }
    if (!flags.containsKey("output_file")) {
      // Different from blaze, this should be mandatory
      throw new IllegalArgumentException("output_file was not specified");
    }

    return flags;
  }
}
