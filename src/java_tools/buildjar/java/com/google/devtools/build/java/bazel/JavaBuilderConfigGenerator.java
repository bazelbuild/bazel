// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.java.bazel;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.rules.java.JavaToolchainData;
import com.google.devtools.build.lib.rules.java.JavaToolchainDataParser;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

/** Utility class to generate {@link JavaBuilderConfig}. */
public class JavaBuilderConfigGenerator {

  private static void die(String message) {
    System.err.println(message);
    System.err.println("\nThis program is expecting the protocol buffer output of a bazel query");
    System.err.println("containing exactly one java_toolchain target. An example of such output");
    System.err.println("can be obtained by:");
    System.err.println(
        "\bazel query --output=proto "
            + "'kind(java_toolchain, deps(//tools/defaults:java_toolchain))'");
    System.exit(-1);
  }

  public static void main(String[] args) {
    try {
      InputStream stream = System.in;
      if (args.length > 0) {
        stream = new FileInputStream(args[0]);
      }
      ImmutableMap<String, JavaToolchainData> data = JavaToolchainDataParser.parse(stream);
      if (data.size() != 1) {
        die(data.isEmpty() ? "No java_toolchain found!" : "Multiple java_toolchain found!");
      }
      JavaToolchainData first = data.values().asList().get(0);
      String optsString = Joiner.on("\", \"").join(first.getJavacOptions());
      System.out.println("package com.google.devtools.build.java.bazel;");
      System.out.println("public class JavaBuilderJavacOpts {");
      System.out.println(
          "public static final String[] DEFAULT_JAVACOPTS = {\"" + optsString + "\"};");
      System.out.println("}");
    } catch (IOException e) {
      die("Cannot load input file: " + e.getMessage());
    }
  }
}
