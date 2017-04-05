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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;

/**
 * Rule definition for apple_watch1_extension.
 */
public class AppleWatch1ExtensionRule implements RuleDefinition {

  private static final ImmutableSet<String> ALLOWED_DEPS_RULE_CLASSES =
      ImmutableSet.of("objc_library", "objc_import");
  static final String WATCH_APP_DEPS_ATTR  = "app_deps";
  static final String WATCH_EXT_FAMILIES_ATTR = "ext_families";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(ObjcConfiguration.class, AppleConfiguration.class)
        /*<!-- #BLAZE_RULE(apple_watch1_extension).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>.ipa</code>: the extension bundle as an <code>.ipa</code>
             file</li>
         <li><code><var>name</var>.xcodeproj/project.pbxproj</code>: An Xcode project file which
             can be used to develop or build on a Mac.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(ReleaseBundlingSupport.IPA, XcodeSupport.PBXPROJ))
        /* <!-- #BLAZE_RULE(apple_watch1_extension).ATTRIBUTE(binary) -->
        The binary target containing the logic for the watch extension. This must be an
        <code>apple_watch1_extension_binary</code> target.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("binary", LABEL)
            .allowedRuleClasses("apple_watch_extension_binary")
            .allowedFileTypes()
            .mandatory()
            .direct_compile_time_input()
            .cfg(AppleWatch1Extension.MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION))
        /* <!-- #BLAZE_RULE(apple_watch1_extension).ATTRIBUTE(app_deps) -->
        The list of targets whose resources files are bundled together to form final watch
        application bundle.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
       .add(attr(WATCH_APP_DEPS_ATTR, LABEL_LIST)
           .direct_compile_time_input()
           .allowedRuleClasses(ALLOWED_DEPS_RULE_CLASSES)
           .allowedFileTypes()
           .cfg(AppleWatch1Extension.MINIMUM_OS_AND_SPLIT_ARCH_TRANSITION))
       /* <!-- #BLAZE_RULE(apple_watch1_extension).ATTRIBUTE(ext_families) -->
       The device families to which the watch extension is targeted.

       This is known as the <code>TARGETED_DEVICE_FAMILY</code> build setting
       in Xcode project files. It is a list of one or more of the strings
       <code>"iphone"</code> and <code>"ipad"</code>.

       <p>By default this is set to <code>"iphone"</code>. If it is explicitly specified it may not
       be empty.</p>
       <p>The watch application is always built for <code>"watch"</code> for device builds and
       <code>"iphone, watch"</code> for simulator builds.
       <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
       .add(
           attr(WATCH_EXT_FAMILIES_ATTR, STRING_LIST)
               .value(ImmutableList.of(TargetDeviceFamily.IPHONE.getNameInRule())))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_watch1_extension")
        .factoryClass(AppleWatch1Extension.class)
        .ancestors(BaseRuleClasses.BaseRule.class,
            ObjcRuleClasses.XcodegenRule.class,
            ObjcRuleClasses.WatchApplicationBundleRule.class,
            ObjcRuleClasses.WatchExtensionBundleRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = apple_watch1_extension, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule produces an extension bundle for apple watch OS 1 which also contains the watch
application bundle</p>

${IMPLICIT_OUTPUTS}

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
