// Copyright 2014 Google Inc. All rights reserved.
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
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromFunctions;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Rule definition for objc_binary.
 *
 * <p>TODO(bazel-team): Deprecate and phase-out infoplist attribute in favor of the infoplists
 * <em>list</em> attribute.
 */
@BlazeRule(name = "objc_binary",
    factoryClass = ObjcBinary.class,
    ancestors = { ObjcBundleLibraryRule.class })
public class ObjcBinaryRule implements RuleDefinition {
  public static final SafeImplicitOutputsFunction IPA = fromTemplates("%{name}.ipa");

  static final String PROVISIONING_PROFILE_ATTR = "provisioning_profile";

  private static Optional<String> stringAttribute(RuleContext context, String attribute) {
    String value = context.attributes().get(attribute, Type.STRING);
    return value.isEmpty() ? Optional.<String>absent() : Optional.of(value);
  }

  /**
   * Returns the value of the app_icon attribute for the given objc_binary rule, or
   * {@code Optional.absent()} if it is not set.
   */
  public static Optional<String> appIcon(RuleContext context) {
    return stringAttribute(context, "app_icon");
  }

  /**
   * Returns the value of the launch_image attribute for the given objc_binary rule, or
   * {@code Optional.absent()} if it is not set.
   */
  public static Optional<String> launchImage(RuleContext context) {
    return stringAttribute(context, "launch_image");
  }

  @Override
  public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(objc_binary).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>.ipa</code>: the application bundle as an <code>.ipa</code>
             file</li>
         <li><code><var>name</var>.xcodeproj/project.pbxproj: An Xcode project file which can be
             used to develop or build on a Mac.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(fromFunctions(IPA, ObjcRuleClasses.PBXPROJ))
        /* <!-- #BLAZE_RULE(objc_binary).ATTRIBUTE(app_icon) -->
        The name of the application icon, which should be in one of the asset
        catalogs of this target or a (transitive) dependency. In a new project,
        this is initialized to "AppIcon" by Xcode.
        <p>
        If the application icon is not in an asset catalog, do not use this
        attribute. Instead, add a CFBundleIcons entry to the Info.plist file.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("app_icon", STRING))
        /* <!-- #BLAZE_RULE(objc_binary).ATTRIBUTE(launch_image) -->
        The name of the launch image, which should be in one of the asset
        catalogs of this target or a (transitive) dependency. In a new project,
        this is initialized to "LaunchImage" by Xcode.
        <p>
        If the launch image is not in an asset catalog, do not use this
        attribute. Instead, add an appropriately-named image resource to the
        bundle.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("launch_image", STRING))
        /* <!-- #BLAZE_RULE(objc_binary).ATTRIBUTE(entitlements) -->
        The entitlements file required for device builds of this application. See
        <a href="https://developer.apple.com/library/mac/documentation/Miscellaneous/Reference/EntitlementKeyReference/Chapters/AboutEntitlements.html">the apple documentation</a>
        for more information. If absent, the default entitlements from the
        provisioning profile will be used.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("entitlements", LABEL).legacyAllowAnyFileType())
        /* <!-- #BLAZE_RULE(objc_binary).ATTRIBUTE(bundle_id) -->
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
        /* <!-- #BLAZE_RULE(objc_binary).ATTRIBUTE(provisioning_profile) -->
        The provisioning profile (.mobileprovision file) to use when bundling
        the application.
        <p>
        This is only used for non-simulator builds.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(PROVISIONING_PROFILE_ATTR, LABEL)
            .value(env.getLabel("//tools/objc:default_provisioning_profile"))
            .allowedFileTypes(FileType.of(".mobileprovision")))
        // TODO(bazel-team): Consider ways to trim dependencies so that changes to deps of these
        // tools don't trigger all objc_* targets. Right now we check-in deploy jars, but we need a
        // less painful and error-prone way.
        .add(attr("$bundlemerge", LABEL).cfg(HOST).exec()
            .value(env.getLabel("//tools/objc:bundlemerge")))
        .add(attr("$dumpsyms", LABEL).cfg(HOST).exec()
            .value(env.getLabel("//tools/objc:dump_syms")))
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_binary, TYPE = BINARY, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule produces an application bundle by linking one or more Objective-C libraries.</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
