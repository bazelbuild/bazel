// Copyright 2015 Google Inc. All rights reserved.
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
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;

/**
 * Rule definition for ios_application.
 */
@BlazeRule(name = "$ios_application",
    ancestors = { BaseRuleClasses.BaseRule.class,
                  ObjcRuleClasses.ObjcBaseResourcesRule.class,
                  ObjcRuleClasses.ObjcHasInfoplistRule.class,
                  ObjcRuleClasses.ObjcHasEntitlementsRule.class },
    type = RuleClassType.ABSTRACT) // TODO(bazel-team): Add factory once this becomes a real rule.
public class IosApplicationRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE($ios_application).ATTRIBUTE(app_icon) -->
        The name of the application icon, which should be in one of the asset
        catalogs of this target or a (transitive) dependency. In a new project,
        this is initialized to "AppIcon" by Xcode.
        <p>
        If the application icon is not in an asset catalog, do not use this
        attribute. Instead, add a CFBundleIcons entry to the Info.plist file.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("app_icon", STRING))
        /* <!-- #BLAZE_RULE($ios_application).ATTRIBUTE(launch_image) -->
        The name of the launch image, which should be in one of the asset
        catalogs of this target or a (transitive) dependency. In a new project,
        this is initialized to "LaunchImage" by Xcode.
        <p>
        If the launch image is not in an asset catalog, do not use this
        attribute. Instead, add an appropriately-named image resource to the
        bundle.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("launch_image", STRING))
        /* <!-- #BLAZE_RULE($ios_application).ATTRIBUTE(bundle_id) -->
        The bundle ID (reverse-DNS path followed by app name) of the binary. If none is specified, a
        junk value will be used.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("bundle_id", STRING)
            .value(new Attribute.ComputedDefault() {
              @Override
              public Object getDefault(AttributeMap rule) {
                // For tests and similar, we don't want to force people to explicitly specify
                // throw-away data.
                return "example." + rule.getName();
              }
            }))
        /* <!-- #BLAZE_RULE($ios_application).ATTRIBUTE(families) -->
        The device families to which this binary is targeted. This is known as
        the <code>TARGETED_DEVICE_FAMILY</code> build setting in Xcode project
        files. It is a list of one or more of the strings <code>"iphone"</code>
        and <code>"ipad"</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("families", STRING_LIST)
            .value(ImmutableList.of(TargetDeviceFamily.IPHONE.getNameInRule())))
        /* <!-- #BLAZE_RULE($ios_application).ATTRIBUTE(provisioning_profile) -->
        The provisioning profile (.mobileprovision file) to use when bundling
        the application.
        <p>
        This is only used for non-simulator builds.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("provisioning_profile", LABEL)
            .value(env.getLabel("//tools/objc:default_provisioning_profile"))
            .allowedFileTypes(FileType.of(".mobileprovision")))
            // TODO(bazel-team): Consider ways to trim dependencies so that changes to deps of these
            // tools don't trigger all objc_* targets. Right now we check-in deploy jars, but we
            // need a less painful and error-prone way.
        /* <!-- #BLAZE_RULE($ios_application).ATTRIBUTE(binary) -->
        The binary target included in the final bundle.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("binary", LABEL)
            .allowedRuleClasses("objc_binary")
            .allowedFileTypes()
            .mandatory()
            .direct_compile_time_input())
        .add(attr("$bundlemerge", LABEL).cfg(HOST).exec()
            .value(env.getLabel("//tools/objc:bundlemerge")))
        .add(attr("$dumpsyms", LABEL).cfg(HOST).exec()
            .value(env.getLabel("//tools/objc:dump_syms")))
        .build();
  }
}
