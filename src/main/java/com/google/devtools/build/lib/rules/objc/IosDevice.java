// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.base.Strings;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.apple.XcodeConfig;
import com.google.devtools.build.lib.rules.apple.XcodeVersionProperties;

/**
 * Implementation for the "ios_device" rule.
 */
public final class IosDevice implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext context)
      throws InterruptedException, RuleErrorException {
    context.ruleWarning(
        "This rule is deprecated. Please use the new Apple build rules "
            + "(https://github.com/bazelbuild/rules_apple) to build Apple targets.");

    String iosVersionAttribute =
        context.attributes().get(IosDeviceRule.IOS_VERSION_ATTR_NAME, STRING);
    XcodeVersionProperties xcodeVersionProperties =
            context.getPrerequisite(
                IosDeviceRule.XCODE_ATTR_NAME,
                Mode.TARGET,
                XcodeVersionProperties.SKYLARK_CONSTRUCTOR);

    DottedVersion xcodeVersion = null;
    if (xcodeVersionProperties != null && xcodeVersionProperties.getXcodeVersion().isPresent()) {
      xcodeVersion = xcodeVersionProperties.getXcodeVersion().get();
    } else if (XcodeConfig.getXcodeVersion(context) != null) {
      xcodeVersion = XcodeConfig.getXcodeVersion(context);
    }

    DottedVersion iosVersion;
    if (!Strings.isNullOrEmpty(iosVersionAttribute)) {
      iosVersion = DottedVersion.fromString(iosVersionAttribute);
    } else if (xcodeVersionProperties != null) {
      iosVersion = xcodeVersionProperties.getDefaultIosSdkVersion();
    } else {
      iosVersion = XcodeConfig.getSdkVersionForPlatform(context, ApplePlatform.IOS_SIMULATOR);
    }

    IosDeviceProvider provider =
        new IosDeviceProvider.Builder()
            .setType(context.attributes().get(IosDeviceRule.TYPE_ATTR_NAME, STRING))
            .setIosVersion(iosVersion)
            .setLocale(context.attributes().get(IosDeviceRule.LOCALE_ATTR_NAME, STRING))
            .setXcodeVersion(xcodeVersion)
            .build();

    return new RuleConfiguredTargetBuilder(context)
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addNativeDeclaredProvider(provider)
        .add(IosTestSubstitutionProvider.class, provider.iosTestSubstitutionProvider())
        .build();
  }
}
