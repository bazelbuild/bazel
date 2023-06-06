// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.apple.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.apple.XcodeConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchain;

/** Implementation for apple_cc_toolchain rule. */
public class AppleCcToolchain extends CcToolchain {
  public static final String SDK_DIR_KEY = "sdk_dir";
  public static final String SDK_FRAMEWORK_DIR_KEY = "sdk_framework_dir";
  public static final String PLATFORM_DEVELOPER_FRAMEWORK_DIR = "platform_developer_framework_dir";
  public static final String VERSION_MIN_KEY = "version_min";

  @VisibleForTesting
  public static final String XCODE_VERSION_OVERRIDE_VALUE_KEY = "xcode_version_override_value";

  @VisibleForTesting
  public static final String APPLE_SDK_VERSION_OVERRIDE_VALUE_KEY =
      "apple_sdk_version_override_value";

  @VisibleForTesting
  public static final String APPLE_SDK_PLATFORM_VALUE_KEY = "apple_sdk_platform_value";

  @Override
  protected void validateToolchain(RuleContext ruleContext) throws RuleErrorException {
    if (XcodeConfig.getXcodeConfigInfo(ruleContext).getXcodeVersion() == null) {
      ruleContext.throwWithRuleError(
          "Xcode version must be specified to use an Apple CROSSTOOL. If your Xcode version has "
              + "changed recently, verify that \"xcode-select -p\" is correct and then try: "
              + "\"bazel shutdown\" to re-run Xcode configuration");
    }
  }

  @Override
  protected boolean isAppleToolchain() {
    return true;
  }
}
