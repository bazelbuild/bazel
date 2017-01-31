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

package com.google.devtools.build.buildjar.resourcejar;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.CharMatcher;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.Deque;
import javax.annotation.Nullable;

/** A command line options parser for {@link ResourceJarOptions}. */
public class ResourceJarOptionsParser {

  /**
   * Parses command line options into {@link ResourceJarOptions}, expanding any {@code @params}
   * files.
   */
  public static ResourceJarOptions parse(Iterable<String> args) throws IOException {
    ResourceJarOptions.Builder builder = ResourceJarOptions.builder();
    parse(builder, args);
    return builder.build();
  }

  /**
   * Parses command line options into a {@link ResourceJarOptions.Builder}, expanding any
   * {@code @params} files.
   */
  public static void parse(ResourceJarOptions.Builder builder, Iterable<String> args)
      throws IOException {
    Deque<String> argumentDeque = new ArrayDeque<>();
    expandParamsFiles(argumentDeque, args);
    parse(builder, argumentDeque);
  }

  private static final Splitter ARG_SPLITTER =
      Splitter.on(CharMatcher.breakingWhitespace()).omitEmptyStrings().trimResults();

  /**
   * Pre-processes an argument list, expanding arguments of the form {@code @filename} by reading
   * the content of the file and appending whitespace-delimited options to {@code argumentDeque}.
   */
  private static void expandParamsFiles(Deque<String> argumentDeque, Iterable<String> args)
      throws IOException {
    for (String arg : args) {
      if (arg.isEmpty()) {
        continue;
      }
      if (arg.startsWith("@") && !arg.startsWith("@@")) {
        Path paramsPath = Paths.get(arg.substring(1));
        expandParamsFiles(
            argumentDeque, ARG_SPLITTER.split(new String(Files.readAllBytes(paramsPath), UTF_8)));
      } else {
        argumentDeque.addLast(arg);
      }
    }
  }

  private static void parse(ResourceJarOptions.Builder builder, Deque<String> argumentDeque) {
    while (!argumentDeque.isEmpty()) {
      String next = argumentDeque.pollFirst();
      switch (next) {
        case "--output":
          builder.setOutput(readOne(argumentDeque));
          break;
        case "--messages":
          builder.setMessages(readList(argumentDeque));
          break;
        case "--resources":
          builder.setResources(readList(argumentDeque));
          break;
        case "--resource_jars":
          builder.setResourceJars(readList(argumentDeque));
          break;
        case "--classpath_resources":
          builder.setClasspathResources(readList(argumentDeque));
          break;
        default:
          if (next.isEmpty() && !argumentDeque.isEmpty()) {
            throw new IllegalArgumentException("unknown option: " + next);
          }
      }
    }
  }

  /** Returns the value of an option, or {@code null}. */
  @Nullable
  private static String readOne(Deque<String> argumentDeque) {
    if (argumentDeque.isEmpty() || argumentDeque.peekFirst().startsWith("-")) {
      return null;
    }
    return argumentDeque.pollFirst();
  }

  /** Returns a list of option values. */
  private static ImmutableList<String> readList(Deque<String> argumentDeque) {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    while (!argumentDeque.isEmpty() && !argumentDeque.peekFirst().startsWith("--")) {
      result.add(argumentDeque.pollFirst());
    }
    return result.build();
  }
}
