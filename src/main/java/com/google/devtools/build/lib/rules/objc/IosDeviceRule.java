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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;

/**
 * Rule definition for ios_device.
 */
public final class IosDeviceRule implements RuleDefinition {
  static final String IOS_VERSION_ATTR_NAME = "ios_version";
  static final String XCODE_ATTR_NAME = "xcode";
  static final String TYPE_ATTR_NAME = "type";
  static final String LOCALE_ATTR_NAME = "locale";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(AppleConfiguration.class)
        /* <!-- #BLAZE_RULE(ios_device).ATTRIBUTE(ios_version) -->
        The operating system version of the device. This corresponds to the
        <code>simctl</code> runtime. Defaults to the default ios sdk version for the xcode
        version defined in the <code>xcode</code> attribute, or uses the ios sdk version
        configuration value, if <code>xcode</code> is unspecified.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(IOS_VERSION_ATTR_NAME, STRING))
        /* <!-- #BLAZE_RULE(ios_device).ATTRIBUTE(xcode) -->
        The version of xcode to use to run the simulator. Defaults to the xcode version build
        configuration value.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(XCODE_ATTR_NAME, LABEL)
            .allowedRuleClasses("xcode_version")
            .allowedFileTypes())
        /* <!-- #BLAZE_RULE(ios_device).ATTRIBUTE(type) -->
        The hardware type. This corresponds to the <code>simctl</code> device
        type.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(TYPE_ATTR_NAME, STRING)
            .mandatory())
        .add(attr(LOCALE_ATTR_NAME, STRING)
            .undocumented("this is not yet supported by any test runner")
            .value("en"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("ios_device")
        .factoryClass(IosDevice.class)
        .ancestors(BaseRuleClasses.BaseRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = ios_device, TYPE = BINARY, FAMILY = Objective-C) -->

<p><strong>This rule is deprecated.</strong> Please use the new Apple build rules
(<a href="https://github.com/bazelbuild/rules_apple">https://github.com/bazelbuild/rules_apple</a>)
to build Apple targets.</p>

<p>This rule defines an iOS device profile that defines a simulator against
which to run tests.</p>

<!-- #END_BLAZE_RULE -->*/
