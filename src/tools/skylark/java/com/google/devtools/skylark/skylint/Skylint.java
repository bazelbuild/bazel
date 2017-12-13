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
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/** The main class for the skylint binary. */
public class Skylint {
  public static void main(String[] args) {
    Linter linter = new Linter();
    List<Path> paths = new ArrayList<>();
    for (String arg : args) {
      if (arg.equals("--single-file")) {
        linter.setSingleFileMode();
      } else if (arg.startsWith("--disable-categories=")) {
        for (String categoryName : parseArgumentList(arg, "--disable-categories=")) {
          linter.disableCategory(categoryName);
        }
      } else if (arg.startsWith("--disable-checks=")) {
        for (String checkName : parseArgumentList(arg, "--disable-checks=")) {
          linter.disableCheck(checkName);
        }
      } else {
        paths.add(Paths.get(arg));
      }
    }
    boolean issuesFound = false;
    for (Path path : paths) {
      List<Issue> issues;
      try {
        issues = linter.lint(path);
      } catch (IOException e) {
        issuesFound = true;
        if (e instanceof NoSuchFileException) {
          System.err.println("File not found: " + path);
        } else {
          System.err.println("Error trying to read " + path);
          e.printStackTrace();
        }
        continue;
      }
      if (!issues.isEmpty()) {
        issuesFound = true;
        for (Issue issue : issues) {
          System.out.println(issue.prettyPrint(path.toString()));
        }
      }
    }
    System.exit(issuesFound ? 1 : 0);
  }

  /** Removes the prefix from the argument and returns the list of comma-separated items. */
  private static List<String> parseArgumentList(String arg, String prefix) {
    if (!arg.startsWith(prefix)) {
      throw new IllegalArgumentException("Argument doesn't start with prefix " + prefix);
    }
    List<String> list = new ArrayList<>();
    String[] items = arg.substring(prefix.length()).split(",");
    for (String item : items) {
      item = item.trim();
      if (item.isEmpty()) {
        continue;
      }
      list.add(item);
    }
    return list;
  }
}
