// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests Skylark API for {@link PlatformInfo} providers. */
@RunWith(JUnit4.class)
public class PlatformInfoApiTest extends PlatformTestCase {

  @Test
  public void testPlatform() throws Exception {
    constraintBuilder("//foo:basic").addConstraintValue("value1").write();
    platformBuilder("//foo:my_platform").addConstraint("value1").write();
    assertNoEvents();

    PlatformInfo platformInfo = fetchPlatformInfo("//foo:my_platform");
    assertThat(platformInfo).isNotNull();
    ConstraintSettingInfo constraintSetting =
        ConstraintSettingInfo.create(makeLabel("//foo:basic"));
    ConstraintValueInfo constraintValue =
        ConstraintValueInfo.create(constraintSetting, makeLabel("//foo:value1"));
    assertThat(platformInfo.constraints().get(constraintSetting)).isEqualTo(constraintValue);
    assertThat(platformInfo.remoteExecutionProperties()).isEmpty();
  }

  @Test
  public void testPlatform_overlappingConstraintValueError() throws Exception {
    List<String> lines =
        new ImmutableList.Builder<String>()
            .addAll(
                constraintBuilder("//foo:basic")
                    .addConstraintValue("value1")
                    .addConstraintValue("value2")
                    .lines())
            .addAll(
                platformBuilder("//foo:my_platform")
                    .addConstraint("value1")
                    .addConstraint("value2")
                    .lines())
            .build();

    checkError(
        "foo",
        "my_platform",
        "Duplicate constraint values detected: "
            + "constraint_setting //foo:basic has [//foo:value1, //foo:value2]",
        lines.toArray(new String[] {}));
  }

  @Test
  public void testPlatform_remoteExecution() throws Exception {
    platformBuilder("//foo:my_platform").setRemoteExecutionProperties("foo: val1").write();
    assertNoEvents();

    PlatformInfo platformInfo = fetchPlatformInfo("//foo:my_platform");
    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("foo: val1");
  }

  @Test
  public void testPlatform_parent() throws Exception {
    constraintBuilder("//foo:setting1").addConstraintValue("value1").write();
    constraintBuilder("//foo:setting2").addConstraintValue("value2").write();
    platformBuilder("//foo:parent_platform").addConstraint("value1").write();
    platformBuilder("//foo:my_platform")
        .setParent("//foo:parent_platform")
        .addConstraint("value2")
        .write();
    assertNoEvents();

    PlatformInfo platformInfo = fetchPlatformInfo("//foo:my_platform");
    assertThat(platformInfo).isNotNull();
    ConstraintSettingInfo constraintSetting1 =
        ConstraintSettingInfo.create(makeLabel("//foo:setting1"));
    ConstraintValueInfo constraintValue1 =
        ConstraintValueInfo.create(constraintSetting1, makeLabel("//foo:value1"));
    assertThat(platformInfo.constraints().get(constraintSetting1)).isEqualTo(constraintValue1);
    ConstraintSettingInfo constraintSetting2 =
        ConstraintSettingInfo.create(makeLabel("//foo:setting2"));
    ConstraintValueInfo constraintValue2 =
        ConstraintValueInfo.create(constraintSetting2, makeLabel("//foo:value2"));
    assertThat(platformInfo.constraints().get(constraintSetting2)).isEqualTo(constraintValue2);
  }

  @Test
  public void testPlatform_parent_override() throws Exception {
    constraintBuilder("//foo:setting1")
        .addConstraintValue("value1a")
        .addConstraintValue("value1b")
        .write();
    platformBuilder("//foo:parent_platform").addConstraint("value1a").write();
    platformBuilder("//foo:my_platform").addConstraint("value1b").write();
    assertNoEvents();

    PlatformInfo platformInfo = fetchPlatformInfo("//foo:my_platform");
    assertThat(platformInfo).isNotNull();
    ConstraintSettingInfo constraintSetting1 =
        ConstraintSettingInfo.create(makeLabel("//foo:setting1"));
    ConstraintValueInfo constraintValue1 =
        ConstraintValueInfo.create(constraintSetting1, makeLabel("//foo:value1b"));
    assertThat(platformInfo.constraints().get(constraintSetting1)).isEqualTo(constraintValue1);
  }

  @Test
  public void testPlatform_parent_remoteExecProperties_entireOverride() throws Exception {
    platformBuilder("//foo:parent_platform").setRemoteExecutionProperties("parent props").write();
    platformBuilder("//foo:my_platform")
        .setParent("//foo:parent_platform")
        .setRemoteExecutionProperties("child props")
        .write();
    assertNoEvents();

    PlatformInfo platformInfo = fetchPlatformInfo("//foo:my_platform");
    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("child props");
  }

  @Test
  public void testPlatform_parent_remoteExecProperties_includeParent() throws Exception {
    platformBuilder("//foo:parent_platform").setRemoteExecutionProperties("parent props").write();
    platformBuilder("//foo:my_platform")
        .setParent("//foo:parent_platform")
        .setRemoteExecutionProperties("child ({PARENT_REMOTE_EXECUTION_PROPERTIES}) props")
        .write();
    assertNoEvents();

    PlatformInfo platformInfo = fetchPlatformInfo("//foo:my_platform");
    assertThat(platformInfo).isNotNull();
    assertThat(platformInfo.remoteExecutionProperties()).isEqualTo("child (parent props) props");
  }

  @Test
  public void testPlatform_parent_tooManyParentsError() throws Exception {
    List<String> lines =
        new ImmutableList.Builder<String>()
            .addAll(platformBuilder("//foo:parent_platform1").lines())
            .addAll(platformBuilder("//foo:parent_platform2").lines())
            .addAll(
                ImmutableList.of(
                    "platform(name = 'my_platform',\n",
                    "  parents = [\n",
                    "    ':parent_platform1',\n",
                    "    ':parent_platform2',\n",
                    "  ])"))
            .build();

    checkError(
        "foo",
        "my_platform",
        "in parents attribute of platform rule //foo:my_platform: "
            + "parents attribute must have a single value",
        lines.toArray(new String[] {}));
  }

}
