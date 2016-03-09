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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.IpaRule;

/**
 * Rule definition for ios_application.
 */
public class IosApplicationRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(ObjcConfiguration.class, AppleConfiguration.class)
        /*<!-- #BLAZE_RULE(ios_application).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>.ipa</code>: the application bundle as an <code>.ipa</code>
             file
         <li><code><var>name</var>.xcodeproj/project.pbxproj</code>: An Xcode project file which
             can be used to develop or build on a Mac.
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(ReleaseBundlingSupport.IPA, XcodeSupport.PBXPROJ))
        /* <!-- #BLAZE_RULE(ios_application).ATTRIBUTE(binary) -->
        The binary target included in the final bundle.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("binary", LABEL)
            .allowedRuleClasses("objc_binary")
            .allowedFileTypes()
            .mandatory()
            .direct_compile_time_input()
            .cfg(IosApplication.SPLIT_ARCH_TRANSITION))
        /* <!-- #BLAZE_RULE(ios_application).ATTRIBUTE(extensions) -->
        Any extensions to include in the final application.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("extensions", LABEL_LIST)
            .allowedRuleClasses("ios_extension")
            .allowedFileTypes()
            .direct_compile_time_input())
        .add(attr("$runner_script_template", LABEL).cfg(HOST)
            .value(env.getToolsLabel("//tools/objc:ios_runner.sh.mac_template")))
        .add(attr("$is_executable", BOOLEAN).value(true)
            .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("ios_application")
        .factoryClass(IosApplication.class)
        .ancestors(
            BaseRuleClasses.BaseRule.class,
            ObjcRuleClasses.ReleaseBundlingRule.class,
            ObjcRuleClasses.XcodegenRule.class,
            ObjcRuleClasses.SimulatorRule.class,
            IpaRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = ios_application, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule produces an application bundle for iOS.</p>
<p>When running an iOS application using the <code>run</code> command, environment variables that
are prefixed with <code>IOS_</code> will be passed to the launched application, with the prefix
stripped. For example, if you export <code>IOS_ENV=foo</code>, <code>ENV=foo</code> will be
passed to the application.</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
