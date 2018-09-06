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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintCollectionApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintSettingInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformInfoApi;
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

  private Set<Label> collectLabels(ImmutableSet<ConstraintSettingInfoApi> settings) {
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
