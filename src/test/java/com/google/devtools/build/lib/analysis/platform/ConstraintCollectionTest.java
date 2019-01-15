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
  public void testFindMissing() {
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

    ConstraintCollection collection = new ConstraintCollection(ImmutableList.of(value1, value2));
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
  public void testFindMissing_withDefaultValues() {
    ConstraintSettingInfo setting =
        ConstraintSettingInfo.create(
            Label.parseAbsoluteUnchecked("//foo:s"), Label.parseAbsoluteUnchecked("//foo:value1"));
    ConstraintValueInfo value1 =
        ConstraintValueInfo.create(setting, Label.parseAbsoluteUnchecked("//foo:value1"));
    ConstraintValueInfo value2 =
        ConstraintValueInfo.create(setting, Label.parseAbsoluteUnchecked("//foo:value2"));

    ConstraintCollection collection1 = new ConstraintCollection(ImmutableList.of(value1));
    assertThat(collection1.containsAll(ImmutableList.of(value1))).isTrue();
    assertThat(collection1.findMissing(ImmutableList.of(value1))).isEmpty();
    assertThat(collection1.containsAll(ImmutableList.of(value2))).isFalse();
    assertThat(collection1.findMissing(ImmutableList.of(value2))).containsExactly(value2);

    ConstraintCollection collectionWithDefault = new ConstraintCollection(ImmutableList.of());
    assertThat(collectionWithDefault.containsAll(ImmutableList.of(value1))).isTrue();
    assertThat(collectionWithDefault.findMissing(ImmutableList.of(value1))).isEmpty();
    assertThat(collectionWithDefault.containsAll(ImmutableList.of(value2))).isFalse();
    assertThat(collectionWithDefault.findMissing(ImmutableList.of(value2))).containsExactly(value2);
  }

  @Test
  public void testDiff() {
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

    ConstraintCollection collection1 = new ConstraintCollection(ImmutableList.of(value1, value2a));
    ConstraintCollection collection2 = new ConstraintCollection(ImmutableList.of(value1, value2b));
    assertThat(collection1.diff(collection2)).containsExactly(setting2);
    assertThat(collection1.diff(collection2)).containsAllIn(collection2.diff(collection1));
  }
}
