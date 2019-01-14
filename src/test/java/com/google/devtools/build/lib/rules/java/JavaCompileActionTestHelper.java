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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getJavacArguments;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getProcessorpath;

import com.google.devtools.build.buildjar.OptionsParser;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import java.util.List;
import java.util.Set;

/**
 * A collection of utilities for extracting the values of command-line options passed to JavaBuilder
 * from a Java compilation action , for testing.
 */
public final class JavaCompileActionTestHelper {

  public static Set<String> getDirectJars(SpawnAction javac) throws Exception {
    return getOptions(javac).directJars();
  }

  public static List<String> getProcessorNames(SpawnAction javac) throws Exception {
    return getOptions(javac).getProcessorNames();
  }

  public static List<String> getProcessorPath(SpawnAction javac) throws Exception {
    return getProcessorpath(javac);
  }

  public static List<String> getProcessorpath(SpawnAction javac) throws Exception {
    return getOptions(javac).getProcessorPath();
  }

  public static List<String> getJavacOpts(SpawnAction javac) throws Exception {
    return getOptions(javac).getJavacOpts();
  }

  public static List<String> getSourceFiles(SpawnAction javac) throws Exception {
    return getOptions(javac).getSourceFiles();
  }

  public static List<String> getSourceJars(SpawnAction javac) throws Exception {
    return getOptions(javac).getSourceJars();
  }

  public static StrictDepsMode getStrictJavaDepsMode(SpawnAction javac) throws Exception {
    String strictJavaDeps = getOptions(javac).getStrictJavaDeps();
    return strictJavaDeps != null ? StrictDepsMode.valueOf(strictJavaDeps) : StrictDepsMode.OFF;
  }

  public static List<String> getClasspath(SpawnAction javac) throws Exception {
    return getOptions(javac).getClassPath();
  }

  public static Set<String> getCompileTimeDependencyArtifacts(SpawnAction javac) throws Exception {
    return getOptions(javac).getDepsArtifacts();
  }

  public static String getFixDepsTool(SpawnAction javac) throws Exception {
    return getOptions(javac).getFixDepsTool();
  }

  public static List<String> getBootClassPath(SpawnAction javac) throws Exception {
    return getOptions(javac).getBootClassPath();
  }

  public static List<String> getSourcePathEntries(SpawnAction javac) throws Exception {
    return getOptions(javac).getSourcePath();
  }

  public static List<String> getBootclasspath(SpawnAction javac) throws Exception {
    return getOptions(javac).getBootClassPath();
  }

  public static List<String> getExtdir(SpawnAction javac) throws Exception {
    return getOptions(javac).getExtClassPath();
  }

  /** Returns the JavaBuilder command line, up to the main class or deploy jar. */
  public static List<String> getJavacCommand(SpawnAction action) throws Exception {
    List<String> args = action.getCommandLines().allArguments();
    return args.subList(0, mainClassIndex(args));
  }

  /** Returns the JavaBuilder options. */
  public static List<String> getJavacArguments(SpawnAction action) throws Exception {
    List<String> args = action.getCommandLines().allArguments();
    return args.subList(mainClassIndex(args) + 1, args.size());
  }

  // Find the index of the last argument of the JavaBuilder command, and before the first option
  // that is passed to JavaBuilder.
  private static int mainClassIndex(List<String> args) {
    for (int idx = 0; idx < args.size(); idx++) {
      String arg = args.get(idx);
      if (arg.equals("-jar")) {
        return idx + 1;
      }
      if (arg.contains("JavaBuilder") && !arg.endsWith(".jar")) {
        return idx;
      }
    }
    throw new IllegalStateException(args.toString());
  }

  private static OptionsParser getOptions(SpawnAction javac) throws Exception {
    checkArgument(
        javac.getMnemonic().equals("Javac"),
        "expected a Javac action, was %s",
        javac.getMnemonic());
    return new OptionsParser(getJavacArguments(javac));
  }

  private JavaCompileActionTestHelper() {}
}
