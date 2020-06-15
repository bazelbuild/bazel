// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.buildjar.genclass;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;

/** A command line parser for {@link GenClassOptions}. */
public class GenClassOptionsParser {
  public static GenClassOptions parse(Iterable<String> args) {
    Iterator<String> it = args.iterator();
    GenClassOptions.Builder builder = GenClassOptions.builder();

    while (it.hasNext()) {
      String arg = it.next();
      switch (arg) {
        case "--manifest_proto":
          builder.setManifest(readPath(it));
          break;
        case "--class_jar":
          builder.setClassJar(readPath(it));
          break;
        case "--output_jar":
          builder.setOutputJar(readPath(it));
          break;
        case "--temp_dir":
          builder.setTempDir(readPath(it));
          break;
        default:
          throw new IllegalArgumentException(
              String.format("Unexpected argument: '%s' in %s", arg, args));
      }
    }
    return builder.build();
  }

  private static Path readPath(Iterator<String> it) {
    if (!it.hasNext()) {
      throw new IllegalArgumentException(String.format("Expected more arguments"));
    }
    return Paths.get(it.next());
  }
}
