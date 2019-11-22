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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import java.util.List;

/**
 * A tuple containing the information in a single target of the {@code xcode_version} rule.
 * A single target of this rule contains an official version label decided by Apple, a number
 * of supported aliases one might use to reference this version, and various properties of
 * the xcode version (such as default SDK versions).
 *
 * <p>For example, one may want to reference official xcode version 7.0.1 using the "7" or
 * "7.0" aliases. This official version of xcode may have a default supported iOS SDK of
 * 9.0.
 */
@Immutable
public class XcodeVersionRuleData implements TransitiveInfoProvider {
  private final Label label;
  private final DottedVersion version;
  private final XcodeVersionProperties xcodeVersionProperties;
  private final ImmutableList<String> aliases;

  XcodeVersionRuleData(Label label, Rule rule) {
    NonconfigurableAttributeMapper attrMapper =
        NonconfigurableAttributeMapper.of(rule);

    this.label = label;
    DottedVersion xcodeVersion =
        DottedVersion.fromStringUnchecked(
            attrMapper.get(XcodeVersionRule.VERSION_ATTR_NAME, Type.STRING));
    String iosSdkVersionString =
        attrMapper.get(XcodeVersionRule.DEFAULT_IOS_SDK_VERSION_ATTR_NAME, Type.STRING);
    String watchosSdkVersionString =
        attrMapper.get(XcodeVersionRule.DEFAULT_WATCHOS_SDK_VERSION_ATTR_NAME, Type.STRING);
    String tvosSdkVersionString =
        attrMapper.get(XcodeVersionRule.DEFAULT_TVOS_SDK_VERSION_ATTR_NAME, Type.STRING);
    String macosxSdkVersionString =
        attrMapper.get(XcodeVersionRule.DEFAULT_MACOS_SDK_VERSION_ATTR_NAME, Type.STRING);
    this.version = xcodeVersion;
    this.xcodeVersionProperties =
        new XcodeVersionProperties(
            xcodeVersion,
            iosSdkVersionString,
            watchosSdkVersionString,
            tvosSdkVersionString,
            macosxSdkVersionString);
    this.aliases = ImmutableList.copyOf(
        attrMapper.get(XcodeVersionRule.ALIASES_ATTR_NAME, Type.STRING_LIST));
  }

  /**
   * Returns the label of the owning target of this provider.
   */
  public Label getLabel() {
    return label;
  }

  /**
   * Returns the official xcode version the owning {@code xcode_version} target is referencing.
   */
  public DottedVersion getVersion() {
    return version;
  }

  /**
   * Returns the properties of the {@code xcode_version} target's referenced xcode version.
   */
  public XcodeVersionProperties getXcodeVersionProperties() {
    return xcodeVersionProperties;
  }

  /**
   * Returns the accepted string aliases for this xcode version.
   */
  public List<String> getAliases() {
    return aliases;
  }
}
