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

import com.google.testing.junit.junit4.runner.CategoryFilter;
import org.junit.runner.manipulation.Filter;

import javax.annotation.Nullable;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

/**
 * Configuration for the JUnit4 test runner.
 */
class JUnit4Config {
  // VisibleForTesting
  static final String JUNIT_API_VERSION_PROPERTY = "com.google.testing.junit.runner.apiVersion";

  // VisibleForTesting
  static final String SHOULD_INSTALL_SECURITY_MANAGER_PROPERTY
      = "com.google.testing.junit.runner.shouldInstallTestSecurityManager";

  private final boolean testRunnerFailFast;
  private final String testIncludeFilterRegexp;
  private final String testExcludeFilterRegexp;
  private final Filter categoriesFilter;
  @Nullable private final Path xmlOutputPath;
  private final String junitApiVersion;
  private final boolean shouldInstallSecurityManager;

  private static final String XML_OUTPUT_FILE_ENV_VAR = "XML_OUTPUT_FILE";

  public JUnit4Config(
          String testIncludeFilterRegexp,
          String testExcludeFilterRegexp,
          @Nullable String testIncludeCategories,
          @Nullable String testExcludeCategories,
          @Nullable Path outputXmlFilePath) {
    this(
        false,
        testIncludeFilterRegexp,
        testExcludeFilterRegexp,
        testIncludeCategories,
        testExcludeCategories,
        outputXmlFilePath,
        System.getProperties());
  }

  public JUnit4Config(
          boolean testRunnerFailFast,
          String testIncludeFilterRegexp,
          String testExcludeFilterRegexp,
          @Nullable String testIncludeCategories,
          @Nullable String testExcludeCategories) {
    this(
        testRunnerFailFast,
        testIncludeFilterRegexp,
        testExcludeFilterRegexp,
        testIncludeCategories,
        testExcludeCategories,
        null,
        System.getProperties());
  }

  // VisibleForTesting
  JUnit4Config(
          String testIncludeFilterRegexp,
          String testExcludeFilterRegexp,
          @Nullable String testIncludeCategories,
          @Nullable String testExcludeCategories,
          @Nullable Path xmlOutputPath,
          Properties systemProperties) {
    this(false, testIncludeFilterRegexp, testExcludeFilterRegexp, testIncludeCategories, testExcludeCategories, xmlOutputPath, systemProperties);
  }

  private JUnit4Config(
          boolean testRunnerFailFast,
          String testIncludeFilterRegexp,
          String testExcludeFilterRegexp,
          @Nullable String testIncludeCategories,
          @Nullable String testExcludeCategories,
          @Nullable Path xmlOutputPath,
          Properties systemProperties) {
    this.testRunnerFailFast = testRunnerFailFast;
    this.testIncludeFilterRegexp = testIncludeFilterRegexp;
    this.testExcludeFilterRegexp = testExcludeFilterRegexp;
    try {
      this.categoriesFilter = parseCategoriesFilter(
              testIncludeCategories,
              testExcludeCategories
      );
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
    this.xmlOutputPath = xmlOutputPath;
    junitApiVersion = systemProperties.getProperty(JUNIT_API_VERSION_PROPERTY, "1").trim();
    shouldInstallSecurityManager = installSecurityManager(systemProperties);
  }

  private static Filter parseCategoriesFilter(@Nullable String includesValue, @Nullable String excludesValue) throws ClassNotFoundException {
    Set<Class<?>> includes = resolveClassList(includesValue);
    Set<Class<?>> excludes = resolveClassList(excludesValue);
    if (includes.isEmpty() && excludes.isEmpty()) {
      return Filter.ALL;
    }
    return new CategoryFilter(includes, excludes);
  }

  private static Set<Class<?>> resolveClassList(@Nullable String listValue) throws ClassNotFoundException {
    if (listValue == null || "".equals(listValue)) {
      return Collections.emptySet();
    }
    String[] parts = listValue.split(",");
    Set<Class<?>> classes = new HashSet<>();
    List<String> missingClasses = new ArrayList<>();
    for (String className : parts) {
      try {
        classes.add(Class.forName(className));
      } catch (ClassNotFoundException e) {
        missingClasses.add(className);
      }
    }
    if (!missingClasses.isEmpty()) {
      throw new ClassNotFoundException("The following category classes could not be found: " + String.join(",", missingClasses));
    }
    return classes;
  }


  private static boolean installSecurityManager(Properties systemProperties) {
    String securityManager = systemProperties.getProperty("java.security.manager");
    if (securityManager != null) {
      return false; // Don't install over the specified security manager
    }
    return Boolean.valueOf(
        systemProperties.getProperty(SHOULD_INSTALL_SECURITY_MANAGER_PROPERTY, "true"));
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
  @Nullable
  public Path getXmlOutputPath() {
    if (xmlOutputPath == null) {
      String envXmlOutputPath = System.getenv(XML_OUTPUT_FILE_ENV_VAR);
      return
          envXmlOutputPath == null ? null : FileSystems.getDefault().getPath(envXmlOutputPath);
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

    if (apiVersion != 1) {
      throw new IllegalStateException(
          "Unsupported JUnit Runner API version " + JUNIT_API_VERSION_PROPERTY + "="
          + junitApiVersion + " (must be \\\"1\\\")");
    }
    return apiVersion;
  }

  /**
   * Returns the value of the {@code test_runner_fail_fast} option, or <code>false<code/> if
   * it was not specified.
   */
  boolean getTestRunnerFailFast() {
    return testRunnerFailFast;
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

  /**
   * Returns the category filter, or Filter.ALL if the options were not specified.
   */
  public Filter getCategoriesFilter() {
    return categoriesFilter;
  }

}
