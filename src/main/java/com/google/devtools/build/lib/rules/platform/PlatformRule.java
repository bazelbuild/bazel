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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@link Platform}. */
public class PlatformRule implements RuleDefinition {
  public static final String RULE_NAME = "platform";
  public static final String CONSTRAINT_VALUES_ATTR = "constraint_values";
  public static final String PARENTS_PLATFORM_ATTR = "parents";
  public static final String REMOTE_EXECUTION_PROPS_ATTR = "remote_execution_properties";
  public static final String EXEC_PROPS_ATTR = "exec_properties";
  static final String HOST_PLATFORM_ATTR = "host_platform";
  static final String TARGET_PLATFORM_ATTR = "target_platform";
  static final String CPU_CONSTRAINTS_ATTR = "cpu_constraints";
  static final String OS_CONSTRAINTS_ATTR = "os_constraints";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    /* <!-- #BLAZE_RULE(platform).NAME -->
    <!-- #END_BLAZE_RULE.NAME --> */
    return builder
        .advertiseProvider(PlatformInfo.class)

        /* <!-- #BLAZE_RULE(platform).ATTRIBUTE(constraint_values) -->
        The combination of constraint choices that this platform comprises. In order for a platform
        to apply to a given environment, the environment must have at least the values in this list.

        <p>Each <code>constraint_value</code> in this list must be for a different
        <code>constraint_setting</code>. For example, you cannot define a platform that requires the
        cpu architecture to be both <code>@platforms//cpu:x86_64</code> and
        <code>@bazel_tools//platforms:arm</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(CONSTRAINT_VALUES_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(ConstraintValueInfo.PROVIDER.id()))

        /* <!-- #BLAZE_RULE(platform).ATTRIBUTE(parents) -->
        The label of a <code>platform</code> target that this platform should inherit from. Although
        the attribute takes a list, there should be no more than one platform present. Any
        constraint_settings not set directly on this platform will be found in the parent platform.
        See the section on <a href="#platform_inheritance">Platform Inheritance</a> for details.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(PARENTS_PLATFORM_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(PlatformInfo.PROVIDER.id()))

        /* <!-- #BLAZE_RULE(platform).ATTRIBUTE(remote_execution_properties) -->
        DEPRECATED. Please use exec_properties attribute instead.

        A string used to configure a remote execution platform. Actual builds make no attempt to
        interpret this, it is treated as opaque data that can be used by a specific SpawnRunner.
        This can include data from the parent platform's "remote_execution_properties" attribute,
        by using the macro "{PARENT_REMOTE_EXECUTION_PROPERTIES}". See the section on
        <a href="#platform_inheritance">Platform Inheritance</a> for details.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(REMOTE_EXECUTION_PROPS_ATTR, Type.STRING))

        /* <!-- #BLAZE_RULE(platform).ATTRIBUTE(exec_properties) -->
        A map of strings that affect the way actions are executed remotely. Bazel makes no attempt
        to interpret this, it is treated as opaque data that's forwarded via the Platform field of
        the  <a href="https://github.com/bazelbuild/remote-apis">remote execution protocol</a>.

        This includes any data from the parent platform's <code>exec_properties</code> attributes.
        If the child and parent platform define the same keys, the child's values are kept. Any
        keys associated with a value that is an empty string are removed from the dictionary.

        This attribute is a full replacement for the deprecated
        <code>remote_execution_properties</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .override(attr(EXEC_PROPS_ATTR, Type.STRING_DICT))

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
                .mandatoryProviders(ConstraintValueInfo.PROVIDER.id())
                .undocumented("Should only be used by internal packages."))
        // Undocumented. Indicates to the rule which constraint_values to use for automatic CPU
        // mapping.
        .add(
            attr(OS_CONSTRAINTS_ATTR, BuildType.LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .mandatoryProviders(ConstraintValueInfo.PROVIDER.id())
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
/*<!-- #BLAZE_RULE (NAME = platform, FAMILY = Platform)[GENERIC_RULE] -->

<p>This rule defines a new platform -- a named collection of constraint choices (such as cpu
architecture or compiler version) describing an environment in which part of the build may run.
See the <a href="../platforms.html">Platforms</a> page for
more details.

<h4 id="platform_examples">Example</h4>
<p>
  This defines a platform that describes any environment running Linux on ARM.
</p>
<pre class="code">
platform(
    name = "linux_arm",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:arm",
    ],
)
</pre>

<h3 id="platform_inheritance">Platform Inheritance</h4>
<p>
  Platforms may use the <code>parents</code> attribute to specify another platform that they will
  inherit constraint values from. Although the <code>parents</code> attribute takes a list, no
  more than a single value is currently supported, and specifying multiple parents is an error.
</p>

<p>
  When checking for the value of a constraint setting in a platform, first the values directly set
  (via the <code>constraint_values</code> attribute) are checked, and then the constraint values on
  the parent. This continues recursively up the chain of parent platforms. In this manner, any
  values set directly on a platform will override the values set on the parent.
</p>

<p>
  Platforms inherit the <code>exec_properties</code> attribute from the parent platform.
  The dictionary entries in <code>exec_properties</code> of the parent and child platforms
  will be combined.
  If the same key appears in both the parent's and the child's <code>exec_properties</code>,
  the child's value will be used. If the child platform specifies an empty string as a value, the
  corresponding property will be unset.
</p>

<p>
  Platforms can also inherit the (deprecated) <code>remote_execution_properties</code> attribute
  from the parent platform. Note: new code should use <code>exec_properties</code> instead. The
  logic described below is maintained to be compatible with legacy behavior but will be removed
  in the future.

  The logic for setting the <code>remote_execution_platform</code> is as follows when there
  is a parent platform:

  <ol>
    <li>
      If <code>remote_execution_property</code> is not set on the child platform, the parent's
      <code>remote_execution_properties</code> will be used.
    </li>
    <li>
      If <code>remote_execution_property</code> is set on the child platform, and contains the
      literal string </code>{PARENT_REMOTE_EXECUTION_PROPERTIES}</code>, that macro will be
      replaced with the contents of the parent's <code>remote_execution_property</code> attribute.
    </li>
    <li>
      If <code>remote_execution_property</code> is set on the child platform, and does not contain
      the macro, the child's <code>remote_execution_property</code> will be used unchanged.
    </li>
  </ol>
</p>

<p>
  <em>Since <code>remote_execution_properties</code> is deprecated and will be phased out, mixing
  <code>remote_execution_properties</code> and <code>exec_properties</code> in the same
  inheritance chain is not allowed.</em>
  Prefer to use <code>exec_properties</code> over the deprecated
  <code>remote_execution_properties</code>.
</p>

<h4 id="platform_inheritance_examples">Example: Constraint Values</h4>
<pre class="code">
platform(
    name = "parent",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:arm",
    ],
)
platform(
    name = "child_a",
    parents = [":parent"],
    constraint_values = [
        "@platforms//cpu:x86_64",
    ],
)
platform(
    name = "child_b",
    parents = [":parent"],
)
</pre>

<p>
  In this example, the child platforms have the following properties:

  <ul>
    <li>
      <code>child_a</code> has the constraint values <code>@platforms//os:linux</code> (inherited
      from the parent) and <code>@platforms//cpu:x86_64</code> (set directly on the platform).
    </li>
    <li>
      <code>child_b</code> inherits all constraint values from the parent, and doesn't set any of
      its own.
    </li>
  </ul>
</p>

<h4 id="platform_inheritance_exec_examples">Example: Execution properties</h4>
<pre class="code">
platform(
    name = "parent",
    exec_properties = {
      "k1": "v1",
      "k2": "v2",
    },
)
platform(
    name = "child_a",
    parents = [":parent"],
)
platform(
    name = "child_b",
    parents = [":parent"],
    exec_properties = {
      "k1": "child"
    }
)
platform(
    name = "child_c",
    parents = [":parent"],
    exec_properties = {
      "k1": ""
    }
)
platform(
    name = "child_d",
    parents = [":parent"],
    exec_properties = {
      "k3": "v3"
    }
)
</pre>

<p>
  In this example, the child platforms have the following properties:

  <ul>
    <li>
      <code>child_a</code> inherits the "exec_properties" of the parent and does not set its own.
    </li>
    <li>
      <code>child_b</code> inherits the parent's <code>exec_properties</code> and overrides the
      value of <code>k1</code>. Its <code>exec_properties</code> will be:
      <code>{ "k1": "child", "k2": "v2" }</code>.
    </li>
    <li>
      <code>child_c</code> inherits the parent's <code>exec_properties</code> and unsets
      <code>k1</code>. Its <code>exec_properties</code> will be:
      <code>{ "k2": "v2" }</code>.
    </li>
    <li>
      <code>child_d</code> inherits the parent's <code>exec_properties</code> and adds a new
      property. Its <code>exec_properties</code> will be:
      <code>{ "k1": "v1",  "k2": "v2", "k3": "v3" }</code>.
    </li>
  </ul>
</p>

<!-- #END_BLAZE_RULE -->*/
