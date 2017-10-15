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

import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Main class of the linter library.
 *
 * <p>Most users of the linter library should only need to use this class.
 */
public class Linter {

  /** List of all checkers that the linter runs. */
  private static final Checker[] checkers = {
    ControlFlowChecker::check,
    DocstringChecker::check,
    NamingConventionsChecker::check,
    StatementWithoutEffectChecker::check,
    UsageChecker::check
  };

  /** Function to read files (can be changed for testing). */
  private FileContentsReader fileReader = Files::readAllBytes;

  public Linter setFileContentsReader(FileContentsReader reader) {
    this.fileReader = reader;
    return this;
  }

  /**
   * Runs all checkers on the given file.
   *
   * @param path path of the file
   * @return list of issues found in that file
   */
  public List<Issue> lint(Path path) throws IOException {
    String content = new String(fileReader.read(path), StandardCharsets.ISO_8859_1);
    List<Issue> issues = new ArrayList<>();
    BuildFileAST ast =
        BuildFileAST.parseString(
            event -> {
              if (event.getKind() == EventKind.ERROR || event.getKind() == EventKind.WARNING) {
                issues.add(new Issue(event.getMessage(), event.getLocation()));
              }
            },
            content);
    for (Checker checker : checkers) {
      issues.addAll(checker.check(ast));
    }
    issues.sort(Issue::compare);
    return issues;
  }

  /**
   * Interface with a function that reads a file.
   *
   * <p>This is useful because we can use a fake for testing.
   */
  @FunctionalInterface
  public interface FileContentsReader {
    byte[] read(Path path) throws IOException;
  }

  /** A checker analyzes an AST and produces a list of issues. */
  @FunctionalInterface
  public interface Checker {
    List<Issue> check(BuildFileAST ast);
  }
}
