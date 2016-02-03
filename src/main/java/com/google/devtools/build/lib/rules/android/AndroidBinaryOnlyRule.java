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
package com.google.devtools.build.lib.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;

/**
 * Attributes for {@code android_binary} that are not present on {@code android_test}.
 */
public final class AndroidBinaryOnlyRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
    return builder
//       /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(application_id) -->
//       A full Java-language-style package name for the application. The name should be unique.
//       The name may contain uppercase or lowercase letters ('A' through 'Z'), numbers, and
//       underscores ('_'). However, individual package name parts may only start with letters.
//       The package name serves as a unique identifier for the application. It's also the default
//       name for the application process (see the &lt;application&gt; element's process attribute)
//       and the default task affinity of an activity.
//
//       This overrides the value declared in the manifest.
//       <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("application_id", STRING).undocumented("not ready for production use"))
//       /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(version_code) -->
//       An internal version number. This number is used only to determine whether one version is
//       more recent than another, with higher numbers indicating more recent versions. This is not
//       the version number shown to users; that number is set by the version_name attribute.
//       The value must be set as an integer, such as "100". Each successive version must have a
//       higher number.
//       This overrides the value declared in the manifest.
//
//       Subject to <a href="make-variables.html">"Make" variable</a> substitution.
//       Suggested practice is to declare a varrdef and reference it here so that a particular build
//       invocation will be used to generate apks for release.
//       <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("version_code", STRING).undocumented("not ready for production use"))
//       /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(version_name) -->
//       The version number shown to users. The string has no other purpose than to be displayed to
//       users. The version_code attribute holds the significant version number used internally.
//       This overrides the value declared in the manifest.
//
//       Subject to <a href="make-variables.html">"Make" variable</a> substitution.
//       Suggested practice is to declare a varrdef and reference it here so that a particular build
//       invocation will be used to generate apks for release.
//       <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("version_name", STRING).undocumented("not ready for production use"))
        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(nocompress_extensions) -->
        A list of file extension to leave uncompressed in apk.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("nocompress_extensions", STRING_LIST))
        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(resource_configuration_filters) -->
        A list of resource configuration filters, such 'en' that will limit the resources in the
        apk to only the ones in the 'en' configuration.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("resource_configuration_filters", STRING_LIST))
        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(densities) -->
        Densities to filter for when building the apk.
        This will strip out raster drawable resources that would not be loaded by a device with
        the specified screen densities, to reduce APK size. A corresponding compatible-screens
        section will also be added to the manifest if it does not already contain a superset
        listing.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("densities", STRING_LIST))
        .add(attr("$android_manifest_merge_tool", LABEL)
            .cfg(HOST)
            .exec()
            .value(env.getLabel(env.getToolsRepository()
                + AndroidRuleClasses.MANIFEST_MERGE_TOOL_LABEL)))

        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(multidex) -->
        Whether to split code into multiple dex files.
        Possible values:
        <ul>
          <li><code>native</code>: Split code into multiple dex files when the
            dex 64K index limit is exceeded.
            Assumes native platform support for loading multidex classes at
            runtime. <em class="harmful">This only works with Android L and
            newer</em>.</li>
          <li><code>legacy</code>: Split code into multiple dex files when the
            dex 64K index limit is exceeded. Assumes multidex classes are
            loaded through application code (i.e. no platform support).</li>
          <li><code>manual_main_dex</code>: Split code into multiple dex files when the
            dex 64K index limit is exceeded. The content of the main dex file
            needs to be specified by providing a list of classes in a text file
            using the <a href="#android_binary.main_dex_list">main_dex_list</a>.</li>
          <li><code>off</code>: Compile all code to a single dex file, even if
            if exceeds the index limit.</li>
        </ul>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("multidex", STRING)
            .allowedValues(new AllowedValueSet(MultidexMode.getValidValues()))
            .value(MultidexMode.OFF.getAttributeValue()))
        .removeAttribute("data")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$android_binary_only")
        .type(RuleClassType.ABSTRACT)
        .ancestors(AndroidRuleClasses.AndroidBinaryBaseRule.class)
        .build();
  }
}
