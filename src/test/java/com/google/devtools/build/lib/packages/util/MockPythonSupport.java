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

package com.google.devtools.build.lib.packages.util;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;
import java.util.List;

/**
 * Creates mock BUILD files required for the python rules.
 */
public abstract class MockPythonSupport {

  /** Setup the support for building Python. */
  public abstract void setup(MockToolsConfig config) throws IOException;

  /**
   * Setup support for, and return the string label of, a target that can be passed to {@code
   * --python_top} that causes the {@code py_runtime} consumed by Python rules to be the given
   * {@code pyRuntimeLabel}.
   */
  public abstract String createPythonTopEntryPoint(MockToolsConfig config, String pyRuntimeLabel)
      throws IOException;

  /**
   * Defines a file simulating the part of @rules_python//python:defs.bzl that defines macros for
   * native rules.
   */
  public void writeMacroFile(MockToolsConfig config) throws IOException {
    List<String> ruleNames = ImmutableList.of("py_library", "py_binary", "py_test", "py_runtime");
    config.create(TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/py_rules/macros/BUILD", "");

    StringBuilder macros = new StringBuilder();
    for (String ruleName : ruleNames) {
      Joiner.on("\n")
          .appendTo(
              macros,
              "def " + ruleName + "(**attrs):",
              "    if 'tags' in attrs and attrs['tags'] != None:",
              "        attrs['tags'] = attrs['tags'] +"
                  + " ['__PYTHON_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__']",
              "    else:",
              "        attrs['tags'] = ['__PYTHON_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__']",
              "    native." + ruleName + "(**attrs)");
      macros.append("\n");
    }
    config.create(
        TestConstants.TOOLS_REPOSITORY_SCRATCH + "third_party/py_rules/macros/defs.bzl",
        macros.toString());
  }

  /**
   * Returns a line loading the proper wrapper macro (which adds the magic tag required by {@code
   * --incompatible_load_python_rules_from_bzl}) for each of the given {@code ruleNames}.
   *
   * <p>Otherwise returns the empty string.
   */
  public static String getMacroLoadStatementFor(String... ruleNames) {
    Preconditions.checkState(ruleNames.length > 0);
    StringBuilder loadStatement =
        new StringBuilder()
            .append("load('")
            .append(TestConstants.TOOLS_REPOSITORY)
            .append("//third_party/py_rules/macros:defs.bzl', ");
    ImmutableList.Builder<String> quotedRuleNames = ImmutableList.builder();
    for (String ruleName : ruleNames) {
      quotedRuleNames.add(String.format("'%s'", ruleName));
    }
    Joiner.on(",").appendTo(loadStatement, quotedRuleNames.build());
    loadStatement.append(")");
    return loadStatement.toString();
  }

  /** Same as {@link #getMacroLoadStatementFor}, but loads all applicable macros. */
  public static String getMacroLoadStatement() {
    return getMacroLoadStatementFor("py_library", "py_binary", "py_test", "py_runtime");
  }
}
