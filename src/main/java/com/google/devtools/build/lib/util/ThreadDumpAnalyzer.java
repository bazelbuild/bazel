// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Analyzes thread dumps from {@code jcmd Thread.dump_to_file}, or {@link
 * com.google.devtools.build.lib.util.ThreadDumper}.
 *
 * <p>The analyzer groups threads with the same stack trace and sorts them by name.
 */
public final class ThreadDumpAnalyzer {
  private static final Pattern THREAD_PATTERN = Pattern.compile("#(\\d+)\\s\"([^\"]+)\".*");
  private static final Pattern THREAD_STATE_PATTERN =
      Pattern.compile("\\s+-\\s(locked|lock|parking|waiting).*");

  private final Map<String, List<ThreadLine>> threadsPerStackTrace = new HashMap<>();

  private record ThreadLine(String raw, String id, String name, List<String> states) {}

  /**
   * Analyzes the given thread dump from the given input stream and writes the analysis to the given
   * output stream.
   */
  public void analyze(InputStream in, OutputStream out) throws IOException {
    try (var writer = new PrintWriter(out, false, UTF_8)) {
      var reader = new BufferedReader(new InputStreamReader(in, UTF_8));
      while (true) {
        var line = reader.readLine();
        if (line == null) {
          break;
        }

        var threadMatcher = THREAD_PATTERN.matcher(line);
        if (threadMatcher.matches()) {
          var threadLine =
              new ThreadLine(
                  line, threadMatcher.group(1), threadMatcher.group(2), new ArrayList<>());
          if (groupStackTrace(threadLine, reader)) {
            break;
          }
        } else {
          writer.println(line);
        }
      }

      // Sort the threads with the same stack trace by name
      for (var threads : threadsPerStackTrace.values()) {
        threads.sort((a, b) -> a.name().compareTo(b.name()));
      }

      var sortedEntries = new ArrayList<>(threadsPerStackTrace.entrySet());
      // Sort the entries by the first thread's name in the group.
      sortedEntries.sort(Comparator.comparing(x -> x.getValue().getFirst().name()));

      for (var entry : sortedEntries) {
        var stackTrace = entry.getKey();
        var threads = entry.getValue();
        for (var thread : threads) {
          writer.println(thread.raw());
          for (var state : thread.states()) {
            writer.println(state);
          }
        }
        writer.println(stackTrace);
      }
    }
  }

  /**
   * Groups the stack trace of the given thread with other threads having the same stack trace.
   *
   * @return true if reached EOF.
   */
  private boolean groupStackTrace(ThreadLine threadLine, BufferedReader reader) throws IOException {
    StringBuilder sb = new StringBuilder();
    boolean eof = false;
    while (true) {
      var line = reader.readLine();
      if (line == null) {
        eof = true;
        break;
      }

      if (line.isBlank()) {
        break;
      }

      if (THREAD_STATE_PATTERN.matcher(line).matches()) {
        threadLine.states.add(line);
      } else {
        sb.append(line).append(System.lineSeparator());
      }
    }
    var stackTrace = sb.toString();
    var threads = threadsPerStackTrace.computeIfAbsent(stackTrace, t -> new ArrayList<>());
    threads.add(threadLine);
    return eof;
  }
}
