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
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;

/**
 * Attributes for {@code android_binary} that are not present on {@code android_test}.
 */
public final class AndroidBinaryOnlyRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(nocompress_extensions) -->
        A list of file extension to leave uncompressed in apk.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("nocompress_extensions", STRING_LIST))
        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(crunch_png) -->
        Do PNG crunching (or not). This is independent of nine-patch processing, which is always
        done. Currently only supported for local resources (not android_resources).
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("crunch_png", BOOLEAN).value(true))
        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(resource_configuration_filters) -->
        A list of resource configuration filters, such 'en' that will limit the resources in the
        apk to only the ones in the 'en' configuration.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(ResourceFilter.RESOURCE_CONFIGURATION_FILTERS_NAME, STRING_LIST))
        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(shrink_resources) -->
        Whether to perform resource shrinking. Resources that are not used by the binary will be
        removed from the APK. This is only supported for rules using local resources (i.e. the
        <code>manifest</code> and <code>resource_files</code> attributes) and requires ProGuard. It
        operates in mostly the same manner as the Gradle resource shrinker
        (https://developer.android.com/studio/build/shrink-code.html#shrink-resources).
        <p>Notable differences:
        <ul>
          <li>resources in <code>values/</code> will be removed as well as file based resources</li>
          <li>uses <code>strict mode</code> by default</li>
          <li>removing unused ID resources is not supported</li>
        </ul>
        If resource shrinking is enabled, <code><var>name</var>_files/resource_shrinker.log</code>
        will also be generated, detailing the analysis and deletions performed.
        <p>Possible values:
        <ul>
          <li><code>shrink_resources = 1</code>: Turns on Android resource shrinking</li>
          <li><code>shrink_resources = 0</code>: Turns off Android resource shrinking</li>
          <li><code>shrink_resources = -1</code>: Shrinking is controlled by the
              <a href="../user-manual.html#flag--android_resource_shrinking">
              --android_resource_shrinking</a> flag.</li>
        </ul>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("shrink_resources", TRISTATE).value(TriState.AUTO))
        /* <!-- #BLAZE_RULE(android_binary).ATTRIBUTE(densities) -->
        Densities to filter for when building the apk.
        This will strip out raster drawable resources that would not be loaded by a device with
        the specified screen densities, to reduce APK size. A corresponding compatible-screens
        section will also be added to the manifest if it does not already contain a superset
        listing.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(ResourceFilter.DENSITIES_NAME, STRING_LIST))
        .add(attr("$android_manifest_merge_tool", LABEL)
            .cfg(HOST)
            .exec()
            .value(env.getToolsLabel(AndroidRuleClasses.MANIFEST_MERGE_TOOL_LABEL)))
        .removeAttribute("data")
        .advertiseProvider(ApkProvider.class)
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
