// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox.cgroups.controller.v2;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.Streams;
import com.google.common.io.CharSink;
import com.google.common.io.Files;
import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Scanner;

/** Abstract v2 controller. */
abstract class UnifiedController implements Controller {
  @Override
  public boolean isLegacy() {
    return false;
  }

  protected Path getChild(String name) throws IOException {
    File subtree = getPath().resolve("cgroup.subtree_control").toFile();
    File controllers = getPath().resolve("cgroup.controllers").toFile();
    if (subtree.canWrite() && controllers.canRead()) {
      CharSink sink = Files.asCharSink(subtree, StandardCharsets.UTF_8);
      try (Scanner scanner = new Scanner(controllers, UTF_8)) {
        sink.writeLines(Streams.stream(scanner).map(c -> "+" + c), " ");
      }
    }
    Path path = getPath().resolve(name);
    path.toFile().mkdirs();
    return path;
  }
}
