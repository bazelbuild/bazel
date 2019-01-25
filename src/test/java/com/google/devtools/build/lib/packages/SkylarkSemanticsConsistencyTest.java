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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DynamicCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import java.util.Arrays;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the flow of flags from {@link SkylarkSemanticsOptions} to {@link SkylarkSemantics}, and
 * to and from {@code SkylarkSemantics}' serialized representation.
 *
 * <p>When adding a new option, it is trivial to make a transposition error or a copy/paste error.
 * These tests guard against such errors. The following possible bugs are considered:
 *
 * <ul>
 *   <li>If a new option is added to {@code SkylarkSemantics} but not to {@code
 *       SkylarkSemanticsOptions}, or vice versa, then the programmer will either be unable to
 *       implement its behavior, or unable to test it from the command line and add user
 *       documentation. We hope that the programmer notices this on their own.
 *   <li>If {@link SkylarkSemanticsOptions#toSkylarkSemantics} is not updated to set all fields of
 *       {@code SkylarkSemantics}, then it will fail immediately because all fields of {@link
 *       SkylarkSemantics.Builder} are mandatory.
 *   <li>To catch a copy/paste error where the wrong field's data is threaded through {@code
 *       toSkylarkSemantics()} or {@code deserialize(...)}, we repeatedly generate matching random
 *       instances of the input and expected output objects.
 *   <li>The {@link #checkDefaultsMatch} test ensures that there is no divergence between the
 *       default values of the two classes.
 *   <li>There is no test coverage for failing to update the non-generated webpage documentation. So
 *       don't forget that!
 * </ul>
 */
@RunWith(JUnit4.class)
public class SkylarkSemanticsConsistencyTest {

  private static final int NUM_RANDOM_TRIALS = 10;

  /**
   * Checks that a randomly generated {@link SkylarkSemanticsOptions} object can be converted to a
   * {@link SkylarkSemantics} object with the same field values.
   */
  @Test
  public void optionsToSemantics() throws Exception {
    for (int i = 0; i < NUM_RANDOM_TRIALS; i++) {
      long seed = i;
      SkylarkSemanticsOptions options = buildRandomOptions(new Random(seed));
      SkylarkSemantics semantics = buildRandomSemantics(new Random(seed));
      SkylarkSemantics semanticsFromOptions = options.toSkylarkSemantics();
      assertThat(semanticsFromOptions).isEqualTo(semantics);
    }
  }

  /**
   * Checks that a randomly generated {@link SkylarkSemantics} object can be serialized and
   * deserialized to an equivalent object.
   */
  @Test
  public void serializationRoundTrip() throws Exception {
    DynamicCodec codec = new DynamicCodec(buildRandomSemantics(new Random(2)).getClass());
    for (int i = 0; i < NUM_RANDOM_TRIALS; i++) {
      SkylarkSemantics semantics = buildRandomSemantics(new Random(i));
      SkylarkSemantics deserialized =
          (SkylarkSemantics)
          TestUtils.fromBytes(
              new DeserializationContext(ImmutableMap.of()),
              codec,
              TestUtils.toBytes(new SerializationContext(ImmutableMap.of()), codec, semantics));
      assertThat(deserialized).isEqualTo(semantics);
    }
  }

  @Test
  public void checkDefaultsMatch() {
    SkylarkSemanticsOptions defaultOptions = Options.getDefaults(SkylarkSemanticsOptions.class);
    SkylarkSemantics defaultSemantics = SkylarkSemantics.DEFAULT_SEMANTICS;
    SkylarkSemantics semanticsFromOptions = defaultOptions.toSkylarkSemantics();
    assertThat(semanticsFromOptions).isEqualTo(defaultSemantics);
  }

  @Test
  public void canGetBuilderFromInstance() {
    SkylarkSemantics original = SkylarkSemantics.DEFAULT_SEMANTICS;
    assertThat(original.internalSkylarkFlagTestCanary()).isFalse();
    SkylarkSemantics modified = original.toBuilder().internalSkylarkFlagTestCanary(true).build();
    assertThat(modified.internalSkylarkFlagTestCanary()).isTrue();
  }

  /**
   * Constructs a {@link SkylarkSemanticsOptions} object with random fields. Must access {@code
   * rand} using the same sequence of operations (for the same fields) as {@link
   * #buildRandomSemantics}.
   */
  private static SkylarkSemanticsOptions buildRandomOptions(Random rand) throws Exception {
    return parseOptions(
        // <== Add new options here in alphabetic order ==>
        "--check_third_party_targets_have_licenses=" + rand.nextBoolean(),
        "--experimental_build_setting_api=" + rand.nextBoolean(),
        "--experimental_cc_skylark_api_enabled_packages="
            + rand.nextDouble()
            + ","
            + rand.nextDouble(),
        "--experimental_enable_android_migration_apis=" + rand.nextBoolean(),
        "--experimental_enable_repo_mapping=" + rand.nextBoolean(),
        "--experimental_java_common_create_provider_enabled_packages="
            + rand.nextDouble()
            + ","
            + rand.nextDouble(),
        "--experimental_platforms_api=" + rand.nextBoolean(),
        "--experimental_starlark_config_transitions=" + rand.nextBoolean(),
        "--experimental_transition_whitelist_location=" + rand.nextDouble(),
        "--incompatible_bzl_disallow_load_after_statement=" + rand.nextBoolean(),
        "--incompatible_depset_is_not_iterable=" + rand.nextBoolean(),
        "--incompatible_depset_union=" + rand.nextBoolean(),
        "--incompatible_disable_deprecated_attr_params=" + rand.nextBoolean(),
        "--incompatible_disable_objc_provider_resources=" + rand.nextBoolean(),
        "--incompatible_disallow_data_transition=" + rand.nextBoolean(),
        "--incompatible_disallow_dict_plus=" + rand.nextBoolean(),
        "--incompatible_disallow_filetype=" + rand.nextBoolean(),
        "--incompatible_disallow_legacy_javainfo=" + rand.nextBoolean(),
        "--incompatible_disallow_legacy_java_provider=" + rand.nextBoolean(),
        "--incompatible_disallow_load_labels_to_cross_package_boundaries=" + rand.nextBoolean(),
        "--incompatible_disallow_old_style_args_add=" + rand.nextBoolean(),
        "--incompatible_expand_directories=" + rand.nextBoolean(),
        "--incompatible_generate_javacommon_source_jar=" + rand.nextBoolean(),
        "--incompatible_new_actions_api=" + rand.nextBoolean(),
        "--incompatible_no_attr_license=" + rand.nextBoolean(),
        "--incompatible_no_output_attr_default=" + rand.nextBoolean(),
        "--incompatible_no_support_tools_in_action_inputs=" + rand.nextBoolean(),
        "--incompatible_no_target_output_group=" + rand.nextBoolean(),
        "--incompatible_no_transitive_loads=" + rand.nextBoolean(),
        "--incompatible_remap_main_repo=" + rand.nextBoolean(),
        "--incompatible_remove_native_maven_jar=" + rand.nextBoolean(),
        "--incompatible_require_feature_configuration_for_pic=" + rand.nextBoolean(),
        "--incompatible_strict_argument_ordering=" + rand.nextBoolean(),
        "--incompatible_string_is_not_iterable=" + rand.nextBoolean(),
        "--internal_skylark_flag_test_canary=" + rand.nextBoolean());
  }

  /**
   * Constructs a {@link SkylarkSemantics} object with random fields. Must access {@code rand} using
   * the same sequence of operations (for the same fields) as {@link #buildRandomOptions}.
   */
  private static SkylarkSemantics buildRandomSemantics(Random rand) {
    return SkylarkSemantics.builder()
        // <== Add new options here in alphabetic order ==>
        .checkThirdPartyTargetsHaveLicenses(rand.nextBoolean())
        .experimentalBuildSettingApi(rand.nextBoolean())
        .experimentalCcSkylarkApiEnabledPackages(
            ImmutableList.of(String.valueOf(rand.nextDouble()), String.valueOf(rand.nextDouble())))
        .experimentalEnableAndroidMigrationApis(rand.nextBoolean())
        .experimentalEnableRepoMapping(rand.nextBoolean())
        .experimentalJavaCommonCreateProviderEnabledPackages(
            ImmutableList.of(String.valueOf(rand.nextDouble()), String.valueOf(rand.nextDouble())))
        .experimentalPlatformsApi(rand.nextBoolean())
        .experimentalStarlarkConfigTransitions(rand.nextBoolean())
        .experimentalTransitionWhitelistLocation(String.valueOf(rand.nextDouble()))
        .incompatibleBzlDisallowLoadAfterStatement(rand.nextBoolean())
        .incompatibleDepsetIsNotIterable(rand.nextBoolean())
        .incompatibleDepsetUnion(rand.nextBoolean())
        .incompatibleDisableDeprecatedAttrParams(rand.nextBoolean())
        .incompatibleDisableObjcProviderResources(rand.nextBoolean())
        .incompatibleDisallowDataTransition(rand.nextBoolean())
        .incompatibleDisallowDictPlus(rand.nextBoolean())
        .incompatibleDisallowFileType(rand.nextBoolean())
        .incompatibleDisallowLegacyJavaInfo(rand.nextBoolean())
        .incompatibleDisallowLegacyJavaProvider(rand.nextBoolean())
        .incompatibleDisallowLoadLabelsToCrossPackageBoundaries(rand.nextBoolean())
        .incompatibleDisallowOldStyleArgsAdd(rand.nextBoolean())
        .incompatibleExpandDirectories(rand.nextBoolean())
        .incompatibleGenerateJavaCommonSourceJar(rand.nextBoolean())
        .incompatibleNewActionsApi(rand.nextBoolean())
        .incompatibleNoAttrLicense(rand.nextBoolean())
        .incompatibleNoOutputAttrDefault(rand.nextBoolean())
        .incompatibleNoSupportToolsInActionInputs(rand.nextBoolean())
        .incompatibleNoTargetOutputGroup(rand.nextBoolean())
        .incompatibleNoTransitiveLoads(rand.nextBoolean())
        .incompatibleRemapMainRepo(rand.nextBoolean())
        .incompatibleRemoveNativeMavenJar(rand.nextBoolean())
        .incompatibleRequireFeatureConfigurationForPic(rand.nextBoolean())
        .incompatibleStricArgumentOrdering(rand.nextBoolean())
        .incompatibleStringIsNotIterable(rand.nextBoolean())
        .internalSkylarkFlagTestCanary(rand.nextBoolean())
        .build();
  }

  private static SkylarkSemanticsOptions parseOptions(String... args) throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(SkylarkSemanticsOptions.class);
    parser.setAllowResidue(false);
    parser.parse(Arrays.asList(args));
    return parser.getOptions(SkylarkSemanticsOptions.class);
  }
}

