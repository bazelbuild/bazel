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

package com.google.devtools.build.java.turbine;

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

/** A command line options parser for {@link TurbineOptions}. */
public class TurbineOptionsParser {

  /**
   * Parses a list of command-line options into a {@link TurbineOptions}, expanding any
   * {@code @params} files.
   */
  public static TurbineOptions parse(Iterable<String> args) throws IOException {
    Deque<String> argumentDeque = new ArrayDeque<>();
    expandParamsFiles(argumentDeque, args);
    TurbineOptions.Builder builder = TurbineOptions.builder();
    parse(builder, argumentDeque);
    return builder.build();
  }

  private static final Splitter ARG_SPLITTER =
      Splitter.on(CharMatcher.BREAKING_WHITESPACE).omitEmptyStrings().trimResults();

  /**
   * Pre-processes an argument list, expanding arguments of the form {@code @filename}
   * by reading the content of the file and appending whitespace-delimited options to
   * {@code argumentDeque}.
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

  private static void parse(TurbineOptions.Builder builder, Deque<String> argumentDeque) {
    while (!argumentDeque.isEmpty()) {
      String next = argumentDeque.pollFirst();
      switch (next) {
        case "--output":
          builder.setOutput(readOne(argumentDeque));
          break;
        case "--source_jars":
          builder.setSourceJars(readList(argumentDeque));
          break;
        case "--temp_dir":
          builder.setTempDir(readOne(argumentDeque));
          break;
        case "--processors":
          builder.setProcessors(readList(argumentDeque));
          break;
        case "--processorpath":
          builder.addProcessorPathEntries(splitClasspath(readOne(argumentDeque)));
          break;
        case "--classpath":
          builder.addClassPathEntries(splitClasspath(readOne(argumentDeque)));
          break;
        case "--bootclasspath":
          builder.addBootClassPathEntries(splitClasspath(readOne(argumentDeque)));
          break;
        case "--javacopts":
          builder.addAllJavacOpts(readList(argumentDeque));
          break;
        case "--sources":
          builder.addSources(readList(argumentDeque));
          break;
        case "--output_deps":
          builder.setOutputDeps(readOne(argumentDeque));
          break;
        case "--direct_dependency":
          {
            String jar = readOne(argumentDeque);
            String target = readOne(argumentDeque);
            builder.addDirectJarToTarget(jar, target);
            break;
          }
        case "--indirect_dependency":
          {
            String jar = readOne(argumentDeque);
            String target = readOne(argumentDeque);
            builder.addIndirectJarToTarget(jar, target);
            break;
          }
        case "--deps_artifacts":
          builder.addAllDepsArtifacts(readList(argumentDeque));
          break;
        case "--target_label":
          builder.setTargetLabel(readOne(argumentDeque));
          break;
        case "--strict_java_deps":
          builder.setStrictJavaDeps(readOne(argumentDeque));
          break;
        case "--rule_kind":
          builder.setRuleKind(readOne(argumentDeque));
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

  private static final Splitter CLASSPATH_SPLITTER =
      Splitter.on(':').trimResults().omitEmptyStrings();

  private static ImmutableList<String> splitClasspath(String path) {
    ImmutableList.Builder<String> classpath = ImmutableList.builder();
    if (path != null) {
      classpath.addAll(CLASSPATH_SPLITTER.split(path));
    }
    return classpath.build();
  }
}
