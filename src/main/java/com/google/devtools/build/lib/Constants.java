// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

/**
 * Various constants required by Bazel.
 */
public class Constants {
  private Constants() {
  }

  public static final String PRODUCT_NAME = "bazel";
  public static final ImmutableList<String> DEFAULT_PACKAGE_PATH = ImmutableList.of("%workspace%");
  public static final String MAIN_RULE_CLASS_PROVIDER =
      "com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider";
  public static final ImmutableList<String> IGNORED_TEST_WARNING_PREFIXES = ImmutableList.of();
  public static final String DEFAULT_RUNFILES_PREFIX = "";

  public static final String NATIVE_DEPS_LIB_SUFFIX = "_nativedeps";

  public static final ImmutableList<String> WATCHFS_BLACKLIST = ImmutableList.of();

  public static final String PRELUDE_FILE_DEPOT_RELATIVE_PATH = "tools/build_rules/prelude_bazel";

  /**
   * List of common attributes documentation, relative to {@link com.google.devtools.build.docgen}.
   */
  public static final ImmutableList<String> COMMON_ATTRIBUTES_DOCFILES = ImmutableList.of(
      "templates/attributes/common/data.html",
      "templates/attributes/common/deprecation.html",
      "templates/attributes/common/deps.html",
      "templates/attributes/common/distribs.html",
      "templates/attributes/common/features.html",
      "templates/attributes/common/licenses.html",
      "templates/attributes/common/tags.html",
      "templates/attributes/common/testonly.html",
      "templates/attributes/common/visibility.html");

  /**
   * List of documentation for common attributes of *_binary rules, relative to
   * {@link com.google.devtools.build.docgen}.
   */
  public static final ImmutableList<String> BINARY_ATTRIBUTES_DOCFILES = ImmutableList.of(
      "templates/attributes/binary/args.html",
      "templates/attributes/binary/output_licenses.html");

  /**
   * List of documentation for common attributes of *_test rules, relative to
   * {@link com.google.devtools.build.docgen}.
   */
  public static final ImmutableList<String> TEST_ATTRIBUTES_DOCFILES = ImmutableList.of(
      "templates/attributes/test/args.html",
      "templates/attributes/test/size.html",
      "templates/attributes/test/timeout.html",
      "templates/attributes/test/flaky.html");

  /**
   * List of file extensions of which baseline coverage generation is supported.
   */
  public static final ImmutableList<String> BASELINE_COVERAGE_OFFLINE_INSTRUMENTATION_SUFFIXES =
      ImmutableList.<String>of();

  /**
   * Rule classes which specify iOS devices for running tests.
   */
  public static final ImmutableSet<String> IOS_DEVICE_RULE_CLASSES = ImmutableSet.of("ios_device");

  public static final String ANDROID_DEFAULT_SDK = "//external:android/sdk".toString();
  public static final String ANDROID_DEP_PREFIX = "//external:android/".toString();
}
