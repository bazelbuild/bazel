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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Main class of the linter library.
 *
 * <p>Most users of the linter library should only need to use this class.
 */
public class Linter {
  /** Map of all checks and their names. */
  private static final ImmutableMap<String, Check> nameToCheck =
      ImmutableMap.<String, Check>builder()
          .put("bad-operation", BadOperationChecker::check)
          .put("control-flow", ControlFlowChecker::check)
          .put("docstring", DocstringChecker::check)
          .put("load", LoadStatementChecker::check)
          .put("naming", NamingConventionsChecker::check)
          .put("no-effect", StatementWithoutEffectChecker::check)
          .put("usage", UsageChecker::check)
          .build();

  /** Function to read files (can be changed for testing). */
  private FileContentsReader fileReader = Files::readAllBytes;

  private final Set<String> disabledChecks = new LinkedHashSet<>();

  public Linter setFileContentsReader(FileContentsReader reader) {
    this.fileReader = reader;
    return this;
  }

  public Linter disable(String checkName) {
    if (!nameToCheck.containsKey(checkName)) {
      throw new IllegalArgumentException("Unknown check '" + checkName + "' cannot be disabled.");
    }
    disabledChecks.add(checkName);
    return this;
  }

  /**
   * Runs all checks on the given file.
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
    for (Entry<String, Check> entry : nameToCheck.entrySet()) {
      if (disabledChecks.contains(entry.getKey())) {
        continue;
      }
      issues.addAll(entry.getValue().check(ast));
    }
    issues.sort(Issue::compareLocation);
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

  /** Allows to invoke a check. */
  @FunctionalInterface
  public interface Check {
    List<Issue> check(BuildFileAST ast);
  }
}
