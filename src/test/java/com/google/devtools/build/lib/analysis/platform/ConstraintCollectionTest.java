// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.platform;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link ConstraintCollection}. */
@RunWith(JUnit4.class)
public class ConstraintCollectionTest extends BuildViewTestCase {
  @Test
  public void testSetArithmetic() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//foo:s1"));
    ConstraintValueInfo value1 =
        ConstraintValueInfo.create(setting1, Label.parseAbsoluteUnchecked("//foo:value1"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//foo:s2"));
    ConstraintValueInfo value2 =
        ConstraintValueInfo.create(setting2, Label.parseAbsoluteUnchecked("//foo:value2"));
    ConstraintSettingInfo setting3 =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//foo:s3"));
    ConstraintValueInfo value3 =
        ConstraintValueInfo.create(setting3, Label.parseAbsoluteUnchecked("//foo:value3"));

    ConstraintCollection collection =
        ConstraintCollection.builder().addConstraints(value1, value2).build();
    assertThat(collection.containsAll(ImmutableList.of(value1))).isTrue();
    assertThat(collection.findMissing(ImmutableList.of(value1))).isEmpty();
    assertThat(collection.containsAll(ImmutableList.of(value2))).isTrue();
    assertThat(collection.containsAll(ImmutableList.of(value1, value2))).isTrue();
    assertThat(collection.containsAll(ImmutableList.of(value3))).isFalse();
    assertThat(collection.findMissing(ImmutableList.of(value3))).containsExactly(value3);
    assertThat(collection.containsAll(ImmutableList.of(value1, value3))).isFalse();
    assertThat(collection.findMissing(ImmutableList.of(value3))).containsExactly(value3);
  }

  @Test
  public void testSetArithmetic_withDefaultValues() throws Exception {
    ConstraintSettingInfo setting =
        ConstraintSettingInfo.create(
            Label.parseAbsoluteUnchecked("//foo:s"), Label.parseAbsoluteUnchecked("//foo:value1"));
    ConstraintValueInfo value1 =
        ConstraintValueInfo.create(setting, Label.parseAbsoluteUnchecked("//foo:value1"));
    ConstraintValueInfo value2 =
        ConstraintValueInfo.create(setting, Label.parseAbsoluteUnchecked("//foo:value2"));

    ConstraintCollection collection1 =
        ConstraintCollection.builder().addConstraints(value1).build();
    assertThat(collection1.containsAll(ImmutableList.of(value1))).isTrue();
    assertThat(collection1.findMissing(ImmutableList.of(value1))).isEmpty();
    assertThat(collection1.containsAll(ImmutableList.of(value2))).isFalse();
    assertThat(collection1.findMissing(ImmutableList.of(value2))).containsExactly(value2);

    ConstraintCollection collectionWithDefault = ConstraintCollection.builder().build();
    assertThat(collectionWithDefault.containsAll(ImmutableList.of(value1))).isTrue();
    assertThat(collectionWithDefault.findMissing(ImmutableList.of(value1))).isEmpty();
    assertThat(collectionWithDefault.containsAll(ImmutableList.of(value2))).isFalse();
    assertThat(collectionWithDefault.findMissing(ImmutableList.of(value2))).containsExactly(value2);
  }

  @Test
  public void testDiff() throws Exception {
    ConstraintSettingInfo setting1 =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//foo:s1"));
    ConstraintValueInfo value1 =
        ConstraintValueInfo.create(setting1, Label.parseAbsoluteUnchecked("//foo:value1"));
    ConstraintSettingInfo setting2 =
        ConstraintSettingInfo.create(Label.parseAbsoluteUnchecked("//foo:s2"));
    ConstraintValueInfo value2a =
        ConstraintValueInfo.create(setting2, Label.parseAbsoluteUnchecked("//foo:value2a"));
    ConstraintValueInfo value2b =
        ConstraintValueInfo.create(setting2, Label.parseAbsoluteUnchecked("//foo:value2b"));

    ConstraintCollection collection1 =
        ConstraintCollection.builder().addConstraints(value1, value2a).build();
    ConstraintCollection collection2 =
        ConstraintCollection.builder().addConstraints(value1, value2b).build();
    assertThat(collection1.diff(collection2)).containsExactly(setting2);
    assertThat(collection1.diff(collection2)).containsAllIn(collection2.diff(collection1));
  }
}
