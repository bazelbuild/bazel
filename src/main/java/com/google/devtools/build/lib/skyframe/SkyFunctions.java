// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Predicate;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Value types in Skyframe.
 */
public final class SkyFunctions {
  public static final SkyFunctionName PRECOMPUTED = SkyFunctionName.create("PRECOMPUTED");
  public static final SkyFunctionName FILE_STATE = SkyFunctionName.create("FILE_STATE");
  public static final SkyFunctionName DIRECTORY_LISTING_STATE =
      SkyFunctionName.create("DIRECTORY_LISTING_STATE");
  public static final SkyFunctionName FILE_SYMLINK_CYCLE_UNIQUENESS =
      SkyFunctionName.create("FILE_SYMLINK_CYCLE_UNIQUENESS");
  public static final SkyFunctionName FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS =
      SkyFunctionName.create("FILE_SYMLINK_INFINITE_EXPANSION_UNIQUENESS");
  public static final SkyFunctionName FILE = SkyFunctionName.create("FILE");
  public static final SkyFunctionName DIRECTORY_LISTING =
      SkyFunctionName.create("DIRECTORY_LISTING");
  public static final SkyFunctionName PACKAGE_LOOKUP = SkyFunctionName.create("PACKAGE_LOOKUP");
  public static final SkyFunctionName CONTAINING_PACKAGE_LOOKUP =
      SkyFunctionName.create("CONTAINING_PACKAGE_LOOKUP");
  public static final SkyFunctionName AST_FILE_LOOKUP = SkyFunctionName.create("AST_FILE_LOOKUP");
  public static final SkyFunctionName SKYLARK_IMPORTS_LOOKUP =
      SkyFunctionName.create("SKYLARK_IMPORTS_LOOKUP");
  public static final SkyFunctionName SKYLARK_IMPORT_CYCLE =
      SkyFunctionName.create("SKYLARK_IMPORT_CYCLE");
  public static final SkyFunctionName GLOB = SkyFunctionName.create("GLOB");
  public static final SkyFunctionName PACKAGE = SkyFunctionName.create("PACKAGE");
  public static final SkyFunctionName PACKAGE_ERROR = SkyFunctionName.create("PACKAGE_ERROR");
  public static final SkyFunctionName TARGET_MARKER = SkyFunctionName.create("TARGET_MARKER");
  public static final SkyFunctionName TARGET_PATTERN = SkyFunctionName.create("TARGET_PATTERN");
  public static final SkyFunctionName PREPARE_DEPS_OF_PATTERNS =
      SkyFunctionName.create("PREPARE_DEPS_OF_PATTERNS");
  public static final SkyFunctionName PREPARE_DEPS_OF_PATTERN =
      SkyFunctionName.create("PREPARE_DEPS_OF_PATTERN");
  public static final SkyFunctionName PREPARE_DEPS_OF_TARGETS_UNDER_DIRECTORY =
      SkyFunctionName.create("PREPARE_DEPS_OF_TARGETS_UNDER_DIRECTORY");
  public static final SkyFunctionName COLLECT_PACKAGES_UNDER_DIRECTORY =
      SkyFunctionName.create("COLLECT_PACKAGES_UNDER_DIRECTORY");
  public static final SkyFunctionName BLACKLISTED_PACKAGE_PREFIXES =
      SkyFunctionName.create("BLACKLISTED_PACKAGE_PREFIXES");
  public static final SkyFunctionName TEST_SUITE_EXPANSION =
      SkyFunctionName.create("TEST_SUITE_EXPANSION");
  public static final SkyFunctionName TESTS_IN_SUITE = SkyFunctionName.create("TESTS_IN_SUITE");
  public static final SkyFunctionName TARGET_PATTERN_PHASE =
      SkyFunctionName.create("TARGET_PATTERN_PHASE");
  public static final SkyFunctionName RECURSIVE_PKG = SkyFunctionName.create("RECURSIVE_PKG");
  public static final SkyFunctionName TRANSITIVE_TARGET =
      SkyFunctionName.create("TRANSITIVE_TARGET");
  public static final SkyFunctionName TRANSITIVE_TRAVERSAL =
      SkyFunctionName.create("TRANSITIVE_TRAVERSAL");
  public static final SkyFunctionName CONFIGURED_TARGET =
      SkyFunctionName.create("CONFIGURED_TARGET");
  public static final SkyFunctionName POST_CONFIGURED_TARGET =
      SkyFunctionName.create("POST_CONFIGURED_TARGET");
  public static final SkyFunctionName ASPECT = SkyFunctionName.create("ASPECT");
  public static final SkyFunctionName LOAD_SKYLARK_ASPECT =
      SkyFunctionName.create("LOAD_SKYLARK_ASPECT");
  public static final SkyFunctionName TARGET_COMPLETION =
      SkyFunctionName.create("TARGET_COMPLETION");
  public static final SkyFunctionName ASPECT_COMPLETION =
      SkyFunctionName.create("ASPECT_COMPLETION");
  public static final SkyFunctionName TEST_COMPLETION = SkyFunctionName.create("TEST_COMPLETION");
  public static final SkyFunctionName BUILD_CONFIGURATION =
      SkyFunctionName.create("BUILD_CONFIGURATION");
  public static final SkyFunctionName CONFIGURATION_FRAGMENT =
      SkyFunctionName.create("CONFIGURATION_FRAGMENT");
  public static final SkyFunctionName CONFIGURATION_COLLECTION =
      SkyFunctionName.create("CONFIGURATION_COLLECTION");
  public static final SkyFunctionName ARTIFACT = SkyFunctionName.create("ARTIFACT");
  public static final SkyFunctionName ACTION_EXECUTION =
      SkyFunctionName.create("ACTION_EXECUTION");
  public static final SkyFunctionName ACTION_LOOKUP = SkyFunctionName.create("ACTION_LOOKUP");
  public static final SkyFunctionName RECURSIVE_FILESYSTEM_TRAVERSAL =
      SkyFunctionName.create("RECURSIVE_DIRECTORY_TRAVERSAL");
  public static final SkyFunctionName FILESET_ENTRY = SkyFunctionName.create("FILESET_ENTRY");
  public static final SkyFunctionName BUILD_INFO_COLLECTION =
      SkyFunctionName.create("BUILD_INFO_COLLECTION");
  public static final SkyFunctionName BUILD_INFO = SkyFunctionName.create("BUILD_INFO");
  public static final SkyFunctionName WORKSPACE_FILE = SkyFunctionName.create("WORKSPACE_FILE");
  public static final SkyFunctionName COVERAGE_REPORT = SkyFunctionName.create("COVERAGE_REPORT");
  public static final SkyFunctionName REPOSITORY = SkyFunctionName.create("REPOSITORY");
  public static final SkyFunctionName REPOSITORY_DIRECTORY =
      SkyFunctionName.create("REPOSITORY_DIRECTORY");
  public static final SkyFunctionName WORKSPACE_AST = SkyFunctionName.create("WORKSPACE_AST");
  public static final SkyFunctionName EXTERNAL_PACKAGE = SkyFunctionName.create("EXTERNAL_PACKAGE");

  public static Predicate<SkyKey> isSkyFunction(final SkyFunctionName functionName) {
    return new Predicate<SkyKey>() {
      @Override
      public boolean apply(SkyKey key) {
        return key.functionName().equals(functionName);
      }
    };
  }
}
