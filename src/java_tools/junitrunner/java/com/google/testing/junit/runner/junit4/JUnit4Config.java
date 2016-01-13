// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;

import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Properties;

/**
 * Configuration for the JUnit4 test runner.
 */
class JUnit4Config {
  @VisibleForTesting
  static final String JUNIT_API_VERSION_PROPERTY = "com.google.testing.junit.runner.apiVersion";

  private final String testIncludeFilterRegexp;
  private final String testExcludeFilterRegexp;
  private final Optional<Path> xmlOutputPath;
  private final String junitApiVersion;
  private final boolean shouldInstallSecurityManager;

  private static final String XML_OUTPUT_FILE_ENV_VAR = "XML_OUTPUT_FILE";

  public JUnit4Config(
      String testIncludeFilterRegexp,
      String testExcludeFilterRegexp,
      Optional<Path> outputXmlFilePath) {
    this(
        testIncludeFilterRegexp,
        testExcludeFilterRegexp,
        outputXmlFilePath,
        System.getProperties());
  }

  @VisibleForTesting
  JUnit4Config(
      String testIncludeFilterRegexp,
      String testExcludeFilterRegexp,
      Optional<Path> xmlOutputPath,
      Properties systemProperties) {
    this.testIncludeFilterRegexp = testIncludeFilterRegexp;
    this.testExcludeFilterRegexp = testExcludeFilterRegexp;
    this.xmlOutputPath = xmlOutputPath;
    junitApiVersion = systemProperties.getProperty(JUNIT_API_VERSION_PROPERTY, "1").trim();
    shouldInstallSecurityManager = systemProperties.getProperty("java.security.manager") == null;
  }

  /**
   * @return Whether the test security manager should be installed
   */
  public boolean shouldInstallSecurityManager() {
    return shouldInstallSecurityManager;
  }

  /**
   * Returns the XML output path, or null if not specified.
   */
  public Optional<Path> getXmlOutputPath() {
    if (!xmlOutputPath.isPresent()) {
      Optional<String> envXmlOutputPath =
          Optional.fromNullable(System.getenv(XML_OUTPUT_FILE_ENV_VAR));
      return envXmlOutputPath.transform(new Function<String, Path>() {
          @Override
          public Path apply(String path) {
            return FileSystems.getDefault().getPath(path);
          }
        });
    }

    return xmlOutputPath;
  }

  /**
   * Gets the version of the JUnit Runner that the test is expecting.
   * Some features may be enabled or disabled based on this value.
   *
   * @return api version
   * @throws IllegalStateException if the API version is unsupported.
   */
  public int getJUnitRunnerApiVersion() {
    int apiVersion = 0;
    try {
      apiVersion = Integer.parseInt(junitApiVersion);
    } catch (NumberFormatException e) {
      // ignore; handled below
    }

    Preconditions.checkState(apiVersion == 1,
        "Unsupported JUnit Runner API version %s=%s (must be \"1\")", JUNIT_API_VERSION_PROPERTY,
        junitApiVersion);
    return apiVersion;
  }

  /**
   * Returns a regular expression representing an inclusive filter.
   * Only test descriptions that match this regular expression should be run.
   */
  public String getTestIncludeFilterRegexp() {
    return testIncludeFilterRegexp;
  }

  /**
   * Returns a regular expression representing an exclusive filter.
   * Test descriptions that match this regular expression should not be run.
   */
  public String getTestExcludeFilterRegexp() {
    return testExcludeFilterRegexp;
  }
}
