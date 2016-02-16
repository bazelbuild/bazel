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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;

/**
 * Rule definition for objc_binary.
 */
// TODO(bazel-team): Remove bundling functionality (dependency on ApplicationRule, IPA output).
public class ObjcBinaryRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(ObjcConfiguration.class, J2ObjcConfiguration.class,
            AppleConfiguration.class)
        /*<!-- #BLAZE_RULE(objc_binary).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>.ipa</code>: the application bundle as an <code>.ipa</code>
             file</li>
         <li><code><var>name</var>.xcodeproj/project.pbxproj</code>: An Xcode project file which
             can be used to develop or build on a Mac.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(ReleaseBundlingSupport.IPA, XcodeSupport.PBXPROJ))
        // TODO(bazel-team): Remove these when this rule no longer produces a bundle.
        .add(attr("$runner_script_template", LABEL).cfg(HOST)
            .value(env.getToolsLabel("//tools/objc:ios_runner.sh.mac_template")))
        .add(attr("$is_executable", BOOLEAN).value(true)
            .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("objc_binary")
        .factoryClass(ObjcBinary.class)
        .ancestors(BaseRuleClasses.BaseRule.class, ObjcRuleClasses.LinkingRule.class,
            ObjcRuleClasses.XcodegenRule.class, ObjcRuleClasses.ReleaseBundlingRule.class,
            ObjcRuleClasses.SimulatorRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_binary, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule produces one or more Objective-C libraries for bundling in an
<code>ios_application</code>.</p>

<p>Any application-related attributes (infoplist, app_icon, ...) on this rule are deprecated and
you should define them on <code>ios_application</code> instead. They will be removed from
<code>objc_binary</code> soon.</p>

<p>Until the migration to <code>ios_application</code> is complete, this rule requires at least one
source file to be defined in either <code>srcs</code> or <code>non_arc_srcs</code></p>.

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
