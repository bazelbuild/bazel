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

package com.google.devtools.build.lib.runtime;

import com.github.luben.zstd.ZstdInputStream;
import com.google.devtools.build.lib.actions.ExecutionGraph;
import com.google.protobuf.TextFormat;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

/** A simple binary to read an action dump and print it as text proto. */
public final class ExecutionGraphProtoToTextMain {

  public static void main(String[] args) throws Exception {
    File f = new File(args[0]).getAbsoluteFile();
    try (InputStream in = new BufferedInputStream(new ZstdInputStream(new FileInputStream(f)))) {
      ExecutionGraph.Node node;
      while ((node = ExecutionGraph.Node.parseDelimitedFrom(in)) != null) {
        System.out.println(TextFormat.printer().printToString(node));
      }
    }
  }

  private ExecutionGraphProtoToTextMain() {}
}
