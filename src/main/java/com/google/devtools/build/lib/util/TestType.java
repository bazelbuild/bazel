// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

/**
 * Utility to detect if in a test. Typically just {@link #isInTest} can be called to branch on
 * unavoidable test-only behavior (avoiding filesystem access, crashing on errors, etc.).
 *
 * <p>Some integration tests may need to distinguish more fully between shell and Java integration
 * tests, and can thread a {@code TestType} object to the necessary libraries to indicate that.
 */
public enum TestType {
  PRODUCTION(false),
  UNKNOWN_TEST(true),
  JAVA_INTEGRATION(true),
  SHELL_INTEGRATION(true);

  private static final TestType TEST_TYPE = getTestTypeFromEnvVars();

  private final boolean inTest;

  TestType(boolean inTest) {
    this.inTest = inTest;
  }

  private static TestType getTestTypeFromEnvVars() {
    boolean inTest = System.getenv("TEST_TMPDIR") != null;
    boolean inShellIntegrationTest = System.getenv("BAZEL_SHELL_TEST") != null;
    return inShellIntegrationTest ? SHELL_INTEGRATION : inTest ? UNKNOWN_TEST : PRODUCTION;
  }

  public static TestType getTestType() {
    return TEST_TYPE;
  }

  public static boolean isInTest() {
    return getTestType().inTest();
  }

  public boolean inTest() {
    return inTest;
  }
}
