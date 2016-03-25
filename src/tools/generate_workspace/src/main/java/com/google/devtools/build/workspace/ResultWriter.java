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

package com.google.devtools.build.workspace;

import com.google.devtools.build.workspace.maven.Rule;

import java.io.PrintStream;
import java.util.Collection;
import java.util.List;

/**
 * Write a set of rules to a WORKSPACE and BUILD file.
 */
public class ResultWriter {
  private final List<String> sources;
  private final Collection<Rule> rules;

  public ResultWriter(List<String> sources, Collection<Rule> rules) {
    this.sources = sources;
    this.rules = rules;
  }

  /**
   * Writes all resolved dependencies in WORKSPACE file format to the outputStream.
   */
  public void writeWorkspace(PrintStream outputStream) {
    writeHeader(outputStream);
    for (Rule rule : rules) {
      outputStream.println(rule + "\n");
    }
  }

  /**
   * Write library rules to depend on the transitive closure of all of these rules.
   */
  public void writeBuild(PrintStream outputStream) {
    writeHeader(outputStream);
    for (Rule rule : rules) {
      outputStream.println("java_library(");
      outputStream.println("    name = \"" + rule.name() + "\",");
      outputStream.println("    visibility = [\"//visibility:public\"],");
      outputStream.println("    exports = [");
      outputStream.println("        \"@" + rule.name() + "//jar\",");
      for (Rule r : rule.getDependencies()) {
        outputStream.println("        \"@" + r.name() + "//jar\",");
      }
      outputStream.println("    ],");
      outputStream.println(")");
      outputStream.println();
    }
  }

  private void writeHeader(PrintStream outputStream) {
    outputStream.println("# The following dependencies were calculated from:");
    for (String header : sources) {
      outputStream.println("# " + header);
    }
    outputStream.print("\n\n");
  }
}
