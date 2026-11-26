// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skymetal;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skymetal.SkymetalFeature.Stage;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link SkymetalFeature}. */
@RunWith(TestParameterInjector.class)
public final class SkymetalFeatureTest {
  private static final String BASE_FEATURE = "base_feature";
  private static final String BASE_FEATURE_PROP = "skymetal." + BASE_FEATURE;
  private static final String OTHER_BASE_FEATURE = "other_base_feature";
  private static final String DEPENDENT_FEATURE = "dependent_feature";
  private static final String DEPENDENT_FEATURE_PROP = "skymetal." + DEPENDENT_FEATURE;
  private static final ImmutableMap<String, String> EMPTY_PROPS = ImmutableMap.of();

  private final List<LogRecord> logRecords = new ArrayList<>();
  private final Logger featureLogger = Logger.getLogger(SkymetalFeature.class.getName());
  private final Handler logHandler =
      new Handler() {
        @Override
        public void publish(LogRecord record) {
          logRecords.add(record);
        }

        @Override
        public void flush() {}

        @Override
        public void close() {}
      };

  @Before
  public void setUp() {
    logRecords.clear();
    featureLogger.addHandler(logHandler);
  }

  @After
  public void tearDown() {
    featureLogger.removeHandler(logHandler);
  }

  @Test
  public void of_unsetTier_enablesOnlyStable() {
    SkymetalFeature stable = SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, EMPTY_PROPS::get);
    assertThat(stable.name()).isEqualTo(BASE_FEATURE);
    assertThat(stable.stage()).isEqualTo(Stage.STABLE);
    assertThat(stable.isEnabled()).isTrue();

    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.PREVIEW, EMPTY_PROPS::get).isEnabled())
        .isFalse();
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.EXPERIMENTAL, EMPTY_PROPS::get).isEnabled())
        .isFalse();
  }

  @Test
  public void of_explicitTier_followsStageHierarchy() {
    ImmutableMap<String, String> stableTier = ImmutableMap.of("skymetal", "stable");
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, stableTier::get).isEnabled())
        .isTrue();
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.PREVIEW, stableTier::get).isEnabled())
        .isFalse();
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.EXPERIMENTAL, stableTier::get).isEnabled())
        .isFalse();

    ImmutableMap<String, String> previewTier = ImmutableMap.of("skymetal", "preview");
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, previewTier::get).isEnabled())
        .isTrue();
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.PREVIEW, previewTier::get).isEnabled())
        .isTrue();
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.EXPERIMENTAL, previewTier::get).isEnabled())
        .isFalse();

    ImmutableMap<String, String> experimentalTier = ImmutableMap.of("skymetal", "experimental");
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, experimentalTier::get).isEnabled())
        .isTrue();
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.PREVIEW, experimentalTier::get).isEnabled())
        .isTrue();
    assertThat(
            SkymetalFeature.of(BASE_FEATURE, Stage.EXPERIMENTAL, experimentalTier::get).isEnabled())
        .isTrue();
  }

  @Test
  public void of_explicitFeatureOverride_overridesTier() {
    ImmutableMap<String, String> enableInStable =
        ImmutableMap.of("skymetal", "stable", BASE_FEATURE_PROP, "true");
    assertThat(
            SkymetalFeature.of(BASE_FEATURE, Stage.EXPERIMENTAL, enableInStable::get).isEnabled())
        .isTrue();
    assertThat(SkymetalFeature.of(BASE_FEATURE, Stage.PREVIEW, enableInStable::get).isEnabled())
        .isTrue();

    ImmutableMap<String, String> disableInExperimental =
        ImmutableMap.of("skymetal", "experimental", BASE_FEATURE_PROP, "false");
    assertThat(
            SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, disableInExperimental::get).isEnabled())
        .isFalse();
    assertThat(
            SkymetalFeature.of(BASE_FEATURE, Stage.EXPERIMENTAL, disableInExperimental::get)
                .isEnabled())
        .isFalse();
  }

  @Test
  public void of_invalidFeatureOverride_throwsIllegalArgumentException(
      @TestParameter({"", "treu", "1", "TRUE"}) String badValue) {
    ImmutableMap<String, String> props = ImmutableMap.of(BASE_FEATURE_PROP, badValue);

    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class,
            () -> SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, props::get));

    assertThat(expected)
        .hasMessageThat()
        .isEqualTo("Invalid -Dskymetal.base_feature=" + badValue + " (expected: true or false)");
  }

  @Test
  public void of_unknownTier_throwsIllegalArgumentException(
      @TestParameter({"", "invalid_tier", "off", "STABLE"}) String badTier) {
    ImmutableMap<String, String> props = ImmutableMap.of("skymetal", badTier);

    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class,
            () -> SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, props::get));

    assertThat(expected)
        .hasMessageThat()
        .isEqualTo(
            "Unknown -Dskymetal=" + badTier + " (expected: stable, preview, or experimental)");
  }

  @Test
  public void of_unknownTierWithExplicitOverride_stillThrowsIllegalArgumentException() {
    ImmutableMap<String, String> props =
        ImmutableMap.of("skymetal", "invalid_tier", BASE_FEATURE_PROP, "true");

    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class,
            () -> SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, props::get));

    assertThat(expected)
        .hasMessageThat()
        .isEqualTo("Unknown -Dskymetal=invalid_tier (expected: stable, preview, or experimental)");
  }

  @Test
  public void of_enabledDependency_enablesDependent() {
    SkymetalFeature enabledBase = SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, EMPTY_PROPS::get);

    SkymetalFeature dependent =
        SkymetalFeature.of(DEPENDENT_FEATURE, Stage.STABLE, EMPTY_PROPS::get, enabledBase);

    assertThat(dependent.isEnabled()).isTrue();
    assertThat(logRecords).isEmpty();
  }

  @Test
  public void of_disabledDependency_cascadesSilently() {
    ImmutableMap<String, String> props = ImmutableMap.of(BASE_FEATURE_PROP, "false");
    SkymetalFeature disabledBase = SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, props::get);

    SkymetalFeature dependent =
        SkymetalFeature.of(DEPENDENT_FEATURE, Stage.STABLE, props::get, disabledBase);

    assertThat(disabledBase.isEnabled()).isFalse();
    assertThat(dependent.isEnabled()).isFalse();
    assertThat(logRecords).isEmpty();
  }

  @Test
  public void of_stageInversionRollback_cascadesSilently() {
    SkymetalFeature rolledBackBase =
        SkymetalFeature.of(BASE_FEATURE, Stage.PREVIEW, EMPTY_PROPS::get);

    SkymetalFeature dependent =
        SkymetalFeature.of(DEPENDENT_FEATURE, Stage.STABLE, EMPTY_PROPS::get, rolledBackBase);

    assertThat(rolledBackBase.isEnabled()).isFalse();
    assertThat(dependent.isEnabled()).isFalse();
    assertThat(logRecords).isEmpty();
  }

  @Test
  public void of_multipleDependenciesWithOneDisabled_disablesAndWarnsOnExplicitConflict() {
    ImmutableMap<String, String> props =
        ImmutableMap.of(BASE_FEATURE_PROP, "false", DEPENDENT_FEATURE_PROP, "true");
    SkymetalFeature enabledBase = SkymetalFeature.of(OTHER_BASE_FEATURE, Stage.STABLE, props::get);
    SkymetalFeature disabledBase = SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, props::get);

    SkymetalFeature dependent =
        SkymetalFeature.of(DEPENDENT_FEATURE, Stage.STABLE, props::get, enabledBase, disabledBase);

    assertThat(enabledBase.isEnabled()).isTrue();
    assertThat(dependent.isEnabled()).isFalse();
    assertThat(logRecords).hasSize(1);
    assertThat(logRecords.get(0).getLevel()).isEqualTo(Level.WARNING);
    assertThat(logRecords.get(0).getMessage())
        .isEqualTo(
            "Cannot enable Skymetal feature 'dependent_feature' (-Dskymetal.dependent_feature=true)"
                + " because prerequisite 'base_feature' is disabled");
  }

  @Test
  public void of_explicitlyDisabledWithDisabledDependency_returnsFalseWithoutWarning() {
    ImmutableMap<String, String> props =
        ImmutableMap.of(BASE_FEATURE_PROP, "false", DEPENDENT_FEATURE_PROP, "false");
    SkymetalFeature disabledBase = SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, props::get);

    SkymetalFeature dependent =
        SkymetalFeature.of(DEPENDENT_FEATURE, Stage.STABLE, props::get, disabledBase);

    assertThat(dependent.isEnabled()).isFalse();
    assertThat(logRecords).isEmpty();
  }

  @Test
  public void of_explicitlyEnabledWithDisabledDependency_logsWarningAndReturnsFalse() {
    ImmutableMap<String, String> props =
        ImmutableMap.of(BASE_FEATURE_PROP, "false", DEPENDENT_FEATURE_PROP, "true");
    SkymetalFeature disabledBase = SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, props::get);

    SkymetalFeature dependent =
        SkymetalFeature.of(DEPENDENT_FEATURE, Stage.STABLE, props::get, disabledBase);

    assertThat(dependent.isEnabled()).isFalse();
    assertThat(logRecords).hasSize(1);
    LogRecord warning = logRecords.get(0);
    assertThat(warning.getLevel()).isEqualTo(Level.WARNING);
    assertThat(warning.getMessage())
        .isEqualTo(
            "Cannot enable Skymetal feature 'dependent_feature' (-Dskymetal.dependent_feature=true)"
                + " because prerequisite 'base_feature' is disabled");
  }

  @Test
  public void of_emptyFeatureName_throwsIllegalArgumentException() {
    assertThrows(
        IllegalArgumentException.class,
        () -> SkymetalFeature.of("", Stage.STABLE, EMPTY_PROPS::get));
    assertThrows(IllegalArgumentException.class, () -> new SkymetalFeature("", Stage.STABLE, true));
  }

  @Test
  public void of_nullArguments_throwsNullPointerException() {
    assertThrows(
        NullPointerException.class, () -> SkymetalFeature.of(null, Stage.STABLE, EMPTY_PROPS::get));
    assertThrows(
        NullPointerException.class, () -> SkymetalFeature.of(BASE_FEATURE, null, EMPTY_PROPS::get));
    assertThrows(
        NullPointerException.class,
        () -> SkymetalFeature.of(BASE_FEATURE, Stage.STABLE, (Function<String, String>) null));
    assertThrows(
        NullPointerException.class,
        () ->
            SkymetalFeature.of(
                BASE_FEATURE, Stage.STABLE, EMPTY_PROPS::get, (SkymetalFeature[]) null));
    assertThrows(
        NullPointerException.class,
        () ->
            SkymetalFeature.of(
                BASE_FEATURE, Stage.STABLE, EMPTY_PROPS::get, (SkymetalFeature) null));
  }
}
