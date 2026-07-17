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

package com.google.devtools.build.importdeps;

import com.google.devtools.build.lib.view.proto.Deps.Dependencies;
import com.google.protobuf.TextFormat;
import java.io.BufferedInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

/** DumpProto pretty-prints a {@code .jdeps} proto. */
public class DumpProto {
  public static void main(String[] args) throws Exception {
    try (BufferedInputStream bis =
        new BufferedInputStream(Files.newInputStream(Paths.get(args[0])))) {
      System.out.print(TextFormat.printer().printToString(Dependencies.parseFrom(bis)));
    }
  }
}
