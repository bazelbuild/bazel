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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Predicate;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Value types in Skyframe.
 */
public final class SkyFunctions {
  public static final SkyFunctionName BUILD_VARIABLE = new SkyFunctionName("BUILD_VARIABLE", false);
  public static final SkyFunctionName FILE_STATE = new SkyFunctionName("FILE_STATE", false);
  public static final SkyFunctionName DIRECTORY_LISTING_STATE =
      new SkyFunctionName("DIRECTORY_LISTING_STATE", false);
  public static final SkyFunctionName FILE_SYMLINK_CYCLE_UNIQUENESS =
      SkyFunctionName.computed("FILE_SYMLINK_CYCLE_UNIQUENESS_NODE");
  public static final SkyFunctionName FILE = SkyFunctionName.computed("FILE");
  public static final SkyFunctionName DIRECTORY_LISTING =
      SkyFunctionName.computed("DIRECTORY_LISTING");
  public static final SkyFunctionName PACKAGE_LOOKUP = SkyFunctionName.computed("PACKAGE_LOOKUP");
  public static final SkyFunctionName CONTAINING_PACKAGE_LOOKUP =
      SkyFunctionName.computed("CONTAINING_PACKAGE_LOOKUP");
  public static final SkyFunctionName AST_LOOKUP = SkyFunctionName.computed("AST_LOOKUP");
  public static final SkyFunctionName SKYLARK_IMPORTS_LOOKUP =
      SkyFunctionName.computed("SKYLARK_IMPORTS_LOOKUP");
  public static final SkyFunctionName GLOB = SkyFunctionName.computed("GLOB");
  public static final SkyFunctionName PACKAGE = SkyFunctionName.computed("PACKAGE");
  public static final SkyFunctionName TARGET_MARKER = SkyFunctionName.computed("TARGET_MARKER");
  public static final SkyFunctionName TARGET_PATTERN = SkyFunctionName.computed("TARGET_PATTERN");
  public static final SkyFunctionName RECURSIVE_PKG = SkyFunctionName.computed("RECURSIVE_PKG");
  public static final SkyFunctionName TRANSITIVE_TARGET =
      SkyFunctionName.computed("TRANSITIVE_TARGET");
  public static final SkyFunctionName CONFIGURED_TARGET =
      SkyFunctionName.computed("CONFIGURED_TARGET");
  public static final SkyFunctionName POST_CONFIGURED_TARGET =
      SkyFunctionName.computed("POST_CONFIGURED_TARGET");
  public static final SkyFunctionName TARGET_COMPLETION =
      SkyFunctionName.computed("TARGET_COMPLETION");
  public static final SkyFunctionName CONFIGURATION_FRAGMENT =
      SkyFunctionName.computed("CONFIGURATION_FRAGMENT");
  public static final SkyFunctionName CONFIGURATION_COLLECTION =
      SkyFunctionName.computed("CONFIGURATION_COLLECTION");
  public static final SkyFunctionName ARTIFACT = SkyFunctionName.computed("ARTIFACT");
  public static final SkyFunctionName ACTION_EXECUTION =
      SkyFunctionName.computed("ACTION_EXECUTION");
  public static final SkyFunctionName ACTION_LOOKUP = SkyFunctionName.computed("ACTION_LOOKUP");
  public static final SkyFunctionName BUILD_INFO_COLLECTION =
      SkyFunctionName.computed("BUILD_INFO_COLLECTION");
  public static final SkyFunctionName BUILD_INFO = SkyFunctionName.computed("BUILD_INFO");

  public static Predicate<SkyKey> isSkyFunction(final SkyFunctionName functionName) {
    return new Predicate<SkyKey>() {
      @Override
      public boolean apply(SkyKey key) {
        return key.functionName() == functionName;
      }
    };
  }
}
