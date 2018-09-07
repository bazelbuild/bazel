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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.SkylarkProvider.SkylarkKey;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintCollectionApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintSettingInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformInfoApi;
import com.google.devtools.build.lib.syntax.SkylarkList;
import java.util.Collection;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests Skylark API for {@link ConstraintCollectionApi} providers. */
@RunWith(JUnit4.class)
public class ConstraintCollectionApiTest extends PlatformInfoApiTest {

  @Test
  public void testConstraintSettings() throws Exception {
    platformBuilder().addConstraint("s1", "value1").addConstraint("s2", "value2").build();

    ConstraintCollectionApi constraintCollection = fetchConstraintCollection();
    assertThat(constraintCollection).isNotNull();

    assertThat(collectLabels(constraintCollection.constraintSettings()))
        .containsExactly(
            Label.parseAbsoluteUnchecked("//foo:s1"), Label.parseAbsoluteUnchecked("//foo:s2"));
  }

  @Test
  public void testGet() throws Exception {
    platformBuilder().addConstraint("s1", "value1").addConstraint("s2", "value2").build();

    ConstraintCollectionApi constraintCollection = fetchConstraintCollection();
    assertThat(constraintCollection).isNotNull();

    ConstraintSettingInfo setting =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//foo:s1"));
    ConstraintValueInfoApi value = constraintCollection.get(setting);
    assertThat(value).isNotNull();
    assertThat(value.label()).isEqualTo(Label.parseAbsoluteUnchecked("//foo:value1"));
  }

  @Test
  public void testGet_starlark() throws Exception {
    platformBuilder().addConstraint("s1", "value1").addConstraint("s2", "value2").build();

    scratch.file("verify/verify.bzl",
        "result = provider()",
        "def _impl(ctx):",
        "  platform = ctx.attr.platform[platform_common.PlatformInfo]",
        "  constraint_setting = ctx.attr.constraint_setting[platform_common.ConstraintSettingInfo]",
        "  constraint_collection = platform.constraints",
        "  value_from_index = constraint_collection[constraint_setting]",
        "  value_from_get = constraint_collection.get(constraint_setting)",
        "  used_constraints = constraint_collection.constraint_settings",
        "  return [result(",
        "    value_from_index = value_from_index,",
        "    value_from_get = value_from_get,",
        "    used_constraints = used_constraints,",
        "  )]",
        "verify = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'platform': attr.label(providers = [platform_common.PlatformInfo]),",
        "    'constraint_setting': attr.label(providers = [platform_common.ConstraintSettingInfo]),",
        "  },",
        ")");
    scratch.file("verify/BUILD",
        "load(':verify.bzl', 'verify')",
        "verify(name = 'verify',",
        "  platform = '//foo:my_platform',",
        "  constraint_setting = '//foo:s1',",
        ")");

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//verify:verify");
    StructImpl info =
        (StructImpl) myRuleTarget.get(
            new SkylarkKey(
                Label.parseAbsolute("//verify:verify.bzl", ImmutableMap.of()), "result"));

    @SuppressWarnings("unchecked")
    ConstraintValueInfo constraintValueFromIndex = (ConstraintValueInfo) info.getValue("value_from_index");
    assertThat(constraintValueFromIndex).isNotNull();
    assertThat(constraintValueFromIndex.label()).isEqualTo(Label.parseAbsoluteUnchecked("//foo:value1"));

    @SuppressWarnings("unchecked")
    ConstraintValueInfo constraintValueFromGet = (ConstraintValueInfo) info.getValue("value_from_get");
    assertThat(constraintValueFromGet).isNotNull();
    assertThat(constraintValueFromGet.label()).isEqualTo(Label.parseAbsoluteUnchecked("//foo:value1"));

    @SuppressWarnings("unchecked")
    SkylarkList<ConstraintSettingInfo> usedConstraints = (SkylarkList<ConstraintSettingInfo>) info.getValue("used_constraints");
    assertThat(usedConstraints).isNotNull();
    assertThat(usedConstraints).containsExactly(ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//foo:s1")), ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//foo:s2")));
  }

  private Set<Label> collectLabels(Collection<ConstraintSettingInfoApi> settings) {
    return settings.stream().map(ConstraintSettingInfoApi::label).collect(Collectors.toSet());
  }

  @Nullable
  private ConstraintCollectionApi fetchConstraintCollection() throws Exception {
    PlatformInfoApi platformInfo = fetchPlatformInfo();
    if (platformInfo == null) {
      return null;
    }
    return platformInfo.constraints();
  }
}
