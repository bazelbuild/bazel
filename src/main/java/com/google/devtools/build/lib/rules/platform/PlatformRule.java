// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.platform;

import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@link Platform}. */
public class PlatformRule implements RuleDefinition {
  public static final String RULE_NAME = "platform";
  public static final String CONSTRAINT_VALUES_ATTR = "constraint_values";
  public static final String PARENTS_PLATFORM_ATTR = "parents";
  public static final String REMOTE_EXECUTION_PROPS_ATTR = "remote_execution_properties";
  static final String HOST_PLATFORM_ATTR = "host_platform";
  static final String TARGET_PLATFORM_ATTR = "target_platform";
  static final String CPU_CONSTRAINTS_ATTR = "cpu_constraints";
  static final String OS_CONSTRAINTS_ATTR = "os_constraints";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    /* <!-- #BLAZE_RULE(platform).NAME -->
    <!-- #END_BLAZE_RULE.NAME --> */
    return builder
        /* <!-- #BLAZE_RULE(platform).ATTRIBUTE(constraint_values) -->
        The combination of constraint choices that this platform comprises. In order for a platform
        to apply to a given environment, the environment must have at least the values in this list.

        <p>Each <code>constraint_value</code> in this list must be for a different
        <code>constraint_setting</code>. For example, you cannot define a platform that requires the
        cpu architecture to be both <code>@bazel_tools//platforms:x86_64</code> and
        <code>@bazel_tools//platforms:arm</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(CONSTRAINT_VALUES_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(ConstraintValueInfo.PROVIDER.id()))

        /* <!-- #BLAZE_RULE(platform).ATTRIBUTE(remote_execution_properties) -->
        A string used to configure a remote execution platform. Actual builds make no attempt to
        interpret this, it is treated as opaque data that can be used by a specific SpawnRunner.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(REMOTE_EXECUTION_PROPS_ATTR, Type.STRING))

        /* <!-- #BLAZE_RULE(platform).ATTRIBUTE(parents) -->
        The label of a <code>platform</code> target that this platform should inherit from. Although
        the attribute takes a list, there should be no more than one platform present. Any
        constraint_settings not set directly on this platform will be found in the parent platform.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(PARENTS_PLATFORM_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(PlatformInfo.PROVIDER.id()))

        // Undocumented. Indicates that this platform should auto-configure the platform constraints
        // based on the current host OS and CPU settings.
        .add(
            attr(HOST_PLATFORM_ATTR, Type.BOOLEAN)
                .value(false)
                .undocumented("Should only be used by internal packages."))
        // Undocumented. Indicates that this platform should auto-configure the platform constraints
        // based on the current OS and CPU settings.
        .add(
            attr(TARGET_PLATFORM_ATTR, Type.BOOLEAN)
                .value(false)
                .undocumented("Should only be used by internal packages."))
        // Undocumented. Indicates to the rule which constraint_values to use for automatic CPU
        // mapping.
        .add(
            attr(CPU_CONSTRAINTS_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(ImmutableList.of(ConstraintValueInfo.PROVIDER.id()))
                .undocumented("Should only be used by internal packages."))
        // Undocumented. Indicates to the rule which constraint_values to use for automatic CPU
        // mapping.
        .add(
            attr(OS_CONSTRAINTS_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(ImmutableList.of(ConstraintValueInfo.PROVIDER.id()))
                .undocumented("Should only be used by internal packages."))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name(RULE_NAME)
        .ancestors(PlatformBaseRule.class)
        .factoryClass(Platform.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = platform, TYPE = OTHER, FAMILY = Platform)[GENERIC_RULE] -->

<p>This rule defines a new platform -- a named collection of constraint choices (such as cpu
architecture or compiler version) describing an environment in which part of the build may run.
See the <a href="https://docs.bazel.build/versions/master/platforms.html">Platforms</a> page for
more details.

<h4 id="platform_examples">Example</h4>
<p>
  This defines a platform that describes any environment running Linux on ARM.
</p>
<pre class="code">
platform(
    name = "linux_arm",
    constraint_values = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:arm",
    ],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
