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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.AppleWatch1ExtensionRule.WATCH_APP_DEPS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.BundlingRule.FAMILIES_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.BundlingRule.INFOPLIST_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.APP_ICON_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.ENTITLEMENTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.LAUNCH_IMAGE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.LAUNCH_STORYBOARD_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_ASSET_CATALOGS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_INFOPLISTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_STORYBOARDS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_INFOPLISTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_PROVISIONING_PROFILE_ATTR;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Preconditions;
import java.io.IOException;
import java.util.Arrays;

/**
 * Represents a pair of rule types, one being a binary and one an bundling type.
 *
 * <p>TODO(bazel-team): Remove this class and refactor tests using it when the binary rule types and
 * bundling rule types are merged.
 */
final class BinaryRuleTypePair {
  private final RuleType binaryType;
  private final RuleType bundlingType;
  private final String bundleDirFormat;
  private final String bundleName;

  /**
   * Creates a rule pair.
   *
   * @param binaryType rule type for the binary half of the pair (e.g. ios_extension_binary)
   * @param bundlingType rule type for wrapper (e.g. ios_extension)
   * @param bundleDirFormat path format for location inside of bundle (e.g. Plugins/%s.appex),
   *     will be substituted with bundle name
   * @param bundleName name of the bundle
   */
  BinaryRuleTypePair(RuleType binaryType, RuleType bundlingType, String bundleDirFormat,
      String bundleName) {
    this.binaryType = binaryType;
    this.bundlingType = bundlingType;
    this.bundleDirFormat = bundleDirFormat;
    this.bundleName = bundleName;
  }

  /**
   * Creates a binary rule pair with bundle name of "x".
   */
  BinaryRuleTypePair(RuleType binaryType, RuleType bundlingType, String bundleDirFormat) {
    this(binaryType, bundlingType, bundleDirFormat, "x");
  }

  /**
   * Returns the name of this bundle.
   */
  String getBundleName() {
    return bundleName;
  }
  
  /**
   * Returns a bundle dir path by combining {@code bundleDirFormat} and {@code bundleName}.
   */
  String getBundleDir() {
    return String.format(bundleDirFormat, bundleName);
  }

  /**
   * Returns the binary type as it appears in {@code BUILD} files, such as {@code objc_binary}.
   */
  RuleType getBinaryRuleType() {
    return binaryType;
  }

  /**
   * Returns the bundling type as it appears in {@code BUILD} files, such as
   * {@code ios_application}.
   */
  RuleType getBundlingRuleType() {
    return bundlingType;
  }

  /**
   * Generates the String necessary to define a bundling and binary rule of these types.
   * This includes an "x" (bundling) and "bin" (binary) target in the given package, setting binary
   * attributes in {@code checkSpecificAttrs} on the binary target, and all other attributes on the
   * bundling target.
   */
  String targets(Scratch scratch, String packageName, String... checkSpecificAttrs)
      throws IOException {
    Preconditions.checkArgument(checkSpecificAttrs.length % 2 == 0,
        "An even number of attribute parameters (kev and value pairs) is required but got: %s",
        Arrays.asList(checkSpecificAttrs));

    ImmutableList.Builder<String> binaryAttributes = new ImmutableList.Builder<>();
    ImmutableList.Builder<String> bundlingAttributes = new ImmutableList.Builder<>();
    bundlingAttributes.add("binary", "':bin'");

    for (int i = 0; i < checkSpecificAttrs.length; i += 2) {
      String attributeName = checkSpecificAttrs[i];
      String value = checkSpecificAttrs[i + 1];
      switch (attributeName) {
        case APP_ICON_ATTR:
        case LAUNCH_IMAGE_ATTR:
        case LAUNCH_STORYBOARD_ATTR:
        case BUNDLE_ID_ATTR:
        case FAMILIES_ATTR:
        case PROVISIONING_PROFILE_ATTR:
        case ENTITLEMENTS_ATTR:
        case INFOPLIST_ATTR:
        case AppleWatch1ExtensionRule.WATCH_EXT_FAMILIES_ATTR:
        case WATCH_EXT_INFOPLISTS_ATTR:
        case WATCH_APP_INFOPLISTS_ATTR:
        case WATCH_APP_PROVISIONING_PROFILE_ATTR:
        case WATCH_EXT_PROVISIONING_PROFILE_ATTR:
        case WATCH_EXT_BUNDLE_ID_ATTR:
        case WATCH_APP_BUNDLE_ID_ATTR:
        case WATCH_APP_STORYBOARDS_ATTR:
        case WATCH_APP_ASSET_CATALOGS_ATTR:
        case WATCH_APP_DEPS_ATTR:
          bundlingAttributes.add(attributeName, value);
          break;
        default:
          binaryAttributes.add(attributeName, value);
      }
    }
    return binaryType.target(scratch, packageName, "bin",
        binaryAttributes.build().toArray(new String[0]))
        + "\n"
        + bundlingType.target(scratch, packageName, "x",
            bundlingAttributes.build().toArray(new String[0]));
  }

  /**
   * Creates targets at //x:x and //x:bin which are the only targets in the BUILD file.
   */
  void scratchTargets(Scratch scratch, String... checkSpecificAttrs) throws IOException {
    scratch.file("x/BUILD", targets(scratch, "x", checkSpecificAttrs));
  }
}
