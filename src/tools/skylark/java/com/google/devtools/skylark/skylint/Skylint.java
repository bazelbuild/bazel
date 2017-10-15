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

package com.google.devtools.skylark.skylint;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/** The main class for the skylint binary. */
public class Skylint {
  public static void main(String[] args) throws IOException {
    Path path = Paths.get(args[0]).toAbsolutePath();
    List<Issue> issues = new Linter().lint(path);
    if (!issues.isEmpty()) {
      System.out.println(path);
      for (Issue issue : issues) {
        System.out.println(issue);
      }
      System.exit(1);
    }
  }
}
