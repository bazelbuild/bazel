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

package com.google.devtools.build.lib.bazel.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabelList;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.java.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.ExperimentalIosTest;
import com.google.devtools.build.lib.rules.objc.IosTest;
import com.google.devtools.build.lib.rules.objc.ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport;
import com.google.devtools.build.lib.rules.objc.XcodeSupport;
import com.google.devtools.build.lib.util.FileType;

import java.util.List;

/**
 * Rule definition for the ios_test rule.
 */
public final class BazelIosTestRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(ObjcConfiguration.class, J2ObjcConfiguration.class,
            AppleConfiguration.class)
        /*<!-- #BLAZE_RULE(ios_test).IMPLICIT_OUTPUTS -->
         <ul>
         <li><code><var>name</var>.ipa</code>: the test bundle as an
         <code>.ipa</code> file
         <li><code><var>name</var>.xcodeproj/project.pbxproj: An Xcode project file which can be
         used to develop or build on a Mac.</li>
         </ul>
         <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(ReleaseBundlingSupport.IPA, XcodeSupport.PBXPROJ))
        .override(
            attr(IosTest.TARGET_DEVICE, LABEL)
                .allowedFileTypes()
                .allowedRuleClasses("ios_device")
                .value(env.getLabel(
                    Constants.TOOLS_REPOSITORY + "//tools/objc/sim_devices:default")))
        /* <!-- #BLAZE_RULE(ios_test).ATTRIBUTE(ios_test_target_device) -->
         The device against how to run the test. If this attribute is defined, the test will run on
         the lab device. Otherwise, the test will run on simulator.
         ${SYNOPSIS}
         <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("ios_test_target_device", LABEL)
                .allowedFileTypes()
                .allowedRuleClasses("ios_lab_device"))
        /* <!-- #BLAZE_RULE(ios_test).ATTRIBUTE(ios_device_arg) -->
         Extra arguments to pass to the <code>ios_test_target_device</code>'s binary. They should be
         in the form KEY=VALUE or simply KEY (check your device's documentation for allowed
         parameters).
         ${SYNOPSIS}
         <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("ios_device_arg", STRING_LIST))
        /* <!-- #BLAZE_RULE(ios_test).ATTRIBUTE(plugins) -->
         Plugins to pass to the test runner.
         ${SYNOPSIS}
         <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("plugins", LABEL_LIST).allowedFileTypes(FileType.of("_deploy.jar")))
        .add(attr("$test_template", LABEL)
            .value(env.getLabel(
                Constants.TOOLS_REPOSITORY + "//tools/objc:ios_test.sh.bazel_template")))
        .add(attr("$test_runner", LABEL)
            .value(env.getLabel(Constants.TOOLS_REPOSITORY + "//tools/objc:testrunner")))
        .add(attr(IosTest.MEMLEAKS_DEP, LABEL)
            .value(env.getLabel("//tools/objc/memleaks:memleaks")))
        .add(attr(IosTest.MEMLEAKS_PLUGIN, LABEL)
            .value(env.getLabel("//tools/objc:memleaks_plugin")))
        .override(attr(":gcov", LABEL_LIST).cfg(HOST)
            .value(new LateBoundLabelList<BuildConfiguration>() {
              @Override
              public List<Label> getDefault(Rule rule, BuildConfiguration configuration) {
                if (!configuration.isCodeCoverageEnabled()) {
                  return ImmutableList.of();
                }
                return ImmutableList.of(
                    configuration.getFragment(ObjcConfiguration.class).getExperimentalGcovLabel());
              }
            }))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("ios_test")
        .type(RuleClassType.TEST)
        .ancestors(BaseRuleClasses.BaseRule.class, BaseRuleClasses.TestBaseRule.class,
            ObjcRuleClasses.IosTestBaseRule.class, ObjcRuleClasses.SimulatorRule.class)
        .factoryClass(ExperimentalIosTest.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = ios_test, TYPE = TEST, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule provides a way to build iOS unit tests written in KIF, GTM and XCTest test frameworks
on both iOS simulator and real devices.
</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
