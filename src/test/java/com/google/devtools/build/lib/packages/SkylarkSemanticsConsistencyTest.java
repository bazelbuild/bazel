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
//

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DynamicCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import java.util.Arrays;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the flow of flags from {@link StarlarkSemanticsOptions} to {@link StarlarkSemantics},
 * and to and from {@code StarlarkSemantics}' serialized representation.
 *
 * <p>When adding a new option, it is trivial to make a transposition error or a copy/paste error.
 * These tests guard against such errors. The following possible bugs are considered:
 *
 * <ul>
 *   <li>If a new option is added to {@code StarlarkSemantics} but not to {@code
 *       StarlarkSemanticsOptions}, or vice versa, then the programmer will either be unable to
 *       implement its behavior, or unable to test it from the command line and add user
 *       documentation. We hope that the programmer notices this on their own.
 *   <li>If {@link StarlarkSemanticsOptions#toSkylarkSemantics} is not updated to set all fields of
 *       {@code StarlarkSemantics}, then it will fail immediately because all fields of {@link
 *       StarlarkSemantics.Builder} are mandatory.
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
   * Checks that a randomly generated {@link StarlarkSemanticsOptions} object can be converted to a
   * {@link StarlarkSemantics} object with the same field values.
   */
  @Test
  public void optionsToSemantics() throws Exception {
    for (int i = 0; i < NUM_RANDOM_TRIALS; i++) {
      long seed = i;
      StarlarkSemanticsOptions options = buildRandomOptions(new Random(seed));
      StarlarkSemantics semantics = buildRandomSemantics(new Random(seed));
      StarlarkSemantics semanticsFromOptions = options.toSkylarkSemantics();
      assertThat(semanticsFromOptions).isEqualTo(semantics);
    }
  }

  /**
   * Checks that a randomly generated {@link StarlarkSemantics} object can be serialized and
   * deserialized to an equivalent object.
   */
  @Test
  public void serializationRoundTrip() throws Exception {
    DynamicCodec codec = new DynamicCodec(buildRandomSemantics(new Random(2)).getClass());
    for (int i = 0; i < NUM_RANDOM_TRIALS; i++) {
      StarlarkSemantics semantics = buildRandomSemantics(new Random(i));
      StarlarkSemantics deserialized =
          (StarlarkSemantics)
              TestUtils.fromBytes(
                  new DeserializationContext(ImmutableMap.of()),
                  codec,
                  TestUtils.toBytes(new SerializationContext(ImmutableMap.of()), codec, semantics));
      assertThat(deserialized).isEqualTo(semantics);
    }
  }

  @Test
  public void checkDefaultsMatch() {
    StarlarkSemanticsOptions defaultOptions = Options.getDefaults(StarlarkSemanticsOptions.class);
    StarlarkSemantics defaultSemantics = StarlarkSemantics.DEFAULT_SEMANTICS;
    StarlarkSemantics semanticsFromOptions = defaultOptions.toSkylarkSemantics();
    assertThat(semanticsFromOptions).isEqualTo(defaultSemantics);
  }

  @Test
  public void canGetBuilderFromInstance() {
    StarlarkSemantics original = StarlarkSemantics.DEFAULT_SEMANTICS;
    assertThat(original.internalSkylarkFlagTestCanary()).isFalse();
    StarlarkSemantics modified = original.toBuilder().internalSkylarkFlagTestCanary(true).build();
    assertThat(modified.internalSkylarkFlagTestCanary()).isTrue();
  }

  /**
   * Constructs a {@link StarlarkSemanticsOptions} object with random fields. Must access {@code
   * rand} using the same sequence of operations (for the same fields) as {@link
   * #buildRandomSemantics}.
   */
  private static StarlarkSemanticsOptions buildRandomOptions(Random rand) throws Exception {
    return parseOptions(
        // <== Add new options here in alphabetic order ==>
        "--debug_depset_depth=" + rand.nextBoolean(),
        "--experimental_action_args=" + rand.nextBoolean(),
        "--experimental_disable_external_package=" + rand.nextBoolean(),
        "--experimental_sibling_repository_layout=" + rand.nextBoolean(),
        "--experimental_allow_incremental_repository_updates=" + rand.nextBoolean(),
        "--experimental_aspect_output_propagation=" + rand.nextBoolean(),
        "--experimental_build_setting_api=" + rand.nextBoolean(),
        "--experimental_cc_skylark_api_enabled_packages="
            + rand.nextDouble()
            + ","
            + rand.nextDouble(),
        "--experimental_enable_android_migration_apis=" + rand.nextBoolean(),
        "--experimental_google_legacy_api=" + rand.nextBoolean(),
        "--experimental_ninja_actions=" + rand.nextBoolean(),
        "--experimental_platforms_api=" + rand.nextBoolean(),
        "--experimental_starlark_config_transitions=" + rand.nextBoolean(),
        "--experimental_starlark_unused_inputs_list=" + rand.nextBoolean(),
        "--incompatible_allow_tags_propagation=" + rand.nextBoolean(), // flag, Java names differ
        "--experimental_cc_shared_library=" + rand.nextBoolean(),
        "--experimental_repo_remote_exec=" + rand.nextBoolean(),
        "--incompatible_always_check_depset_elements=" + rand.nextBoolean(),
        "--incompatible_applicable_licenses=" + rand.nextBoolean(),
        "--incompatible_depset_for_libraries_to_link_getter=" + rand.nextBoolean(),
        "--incompatible_depset_union=" + rand.nextBoolean(),
        "--incompatible_disable_target_provider_fields=" + rand.nextBoolean(),
        "--incompatible_disable_deprecated_attr_params=" + rand.nextBoolean(),
        "--incompatible_disable_depset_items=" + rand.nextBoolean(),
        "--incompatible_disable_third_party_license_checking=" + rand.nextBoolean(),
        "--incompatible_disallow_empty_glob=" + rand.nextBoolean(),
        "--incompatible_disallow_struct_provider_syntax=" + rand.nextBoolean(),
        "--incompatible_do_not_split_linking_cmdline=" + rand.nextBoolean(),
        "--incompatible_linkopts_to_linklibs=" + rand.nextBoolean(),
        "--incompatible_new_actions_api=" + rand.nextBoolean(),
        "--incompatible_no_attr_license=" + rand.nextBoolean(),
        "--incompatible_no_implicit_file_export=" + rand.nextBoolean(),
        "--incompatible_no_output_attr_default=" + rand.nextBoolean(),
        "--incompatible_no_rule_outputs_param=" + rand.nextBoolean(),
        "--incompatible_no_support_tools_in_action_inputs=" + rand.nextBoolean(),
        "--incompatible_no_target_output_group=" + rand.nextBoolean(),
        "--incompatible_restrict_named_params=" + rand.nextBoolean(),
        "--incompatible_run_shell_command_string=" + rand.nextBoolean(),
        "--incompatible_visibility_private_attributes_at_definition=" + rand.nextBoolean(),
        "--incompatible_require_linker_input_cc_api=" + rand.nextBoolean(),
        "--incompatible_restrict_string_escapes=" + rand.nextBoolean(),
        "--incompatible_use_cc_configure_from_rules_cc=" + rand.nextBoolean(),
        "--internal_skylark_flag_test_canary=" + rand.nextBoolean(),
        "--max_computation_steps=" + rand.nextLong(),
        "--record_rule_instantiation_callstack=" + rand.nextBoolean());
  }

  /**
   * Constructs a {@link StarlarkSemantics} object with random fields. Must access {@code rand}
   * using the same sequence of operations (for the same fields) as {@link #buildRandomOptions}.
   */
  private static StarlarkSemantics buildRandomSemantics(Random rand) {
    return StarlarkSemantics.builder()
        // <== Add new options here in alphabetic order ==>
        .debugDepsetDepth(rand.nextBoolean())
        .experimentalActionArgs(rand.nextBoolean())
        .experimentalDisableExternalPackage(rand.nextBoolean())
        .experimentalSiblingRepositoryLayout(rand.nextBoolean())
        .experimentalAllowIncrementalRepositoryUpdates(rand.nextBoolean())
        .experimentalAspectOutputPropagation(rand.nextBoolean())
        .experimentalBuildSettingApi(rand.nextBoolean())
        .experimentalCcSkylarkApiEnabledPackages(
            ImmutableList.of(String.valueOf(rand.nextDouble()), String.valueOf(rand.nextDouble())))
        .experimentalEnableAndroidMigrationApis(rand.nextBoolean())
        .experimentalGoogleLegacyApi(rand.nextBoolean())
        .experimentalNinjaActions(rand.nextBoolean())
        .experimentalPlatformsApi(rand.nextBoolean())
        .experimentalStarlarkConfigTransitions(rand.nextBoolean())
        .experimentalStarlarkUnusedInputsList(rand.nextBoolean())
        .experimentalAllowTagsPropagation(rand.nextBoolean())
        .experimentalCcSharedLibrary(rand.nextBoolean())
        .experimentalRepoRemoteExec(rand.nextBoolean())
        .incompatibleAlwaysCheckDepsetElements(rand.nextBoolean())
        .incompatibleApplicableLicenses(rand.nextBoolean())
        .incompatibleDepsetForLibrariesToLinkGetter(rand.nextBoolean())
        .incompatibleDepsetUnion(rand.nextBoolean())
        .incompatibleDisableTargetProviderFields(rand.nextBoolean())
        .incompatibleDisableDeprecatedAttrParams(rand.nextBoolean())
        .incompatibleDisableDepsetItems(rand.nextBoolean())
        .incompatibleDisableThirdPartyLicenseChecking(rand.nextBoolean())
        .incompatibleDisallowEmptyGlob(rand.nextBoolean())
        .incompatibleDisallowStructProviderSyntax(rand.nextBoolean())
        .incompatibleDoNotSplitLinkingCmdline(rand.nextBoolean())
        .incompatibleLinkoptsToLinkLibs(rand.nextBoolean())
        .incompatibleNewActionsApi(rand.nextBoolean())
        .incompatibleNoAttrLicense(rand.nextBoolean())
        .incompatibleNoImplicitFileExport(rand.nextBoolean())
        .incompatibleNoOutputAttrDefault(rand.nextBoolean())
        .incompatibleNoRuleOutputsParam(rand.nextBoolean())
        .incompatibleNoSupportToolsInActionInputs(rand.nextBoolean())
        .incompatibleNoTargetOutputGroup(rand.nextBoolean())
        .incompatibleRestrictNamedParams(rand.nextBoolean())
        .incompatibleRunShellCommandString(rand.nextBoolean())
        .incompatibleVisibilityPrivateAttributesAtDefinition(rand.nextBoolean())
        .incompatibleRequireLinkerInputCcApi(rand.nextBoolean())
        .incompatibleRestrictStringEscapes(rand.nextBoolean())
        .incompatibleUseCcConfigureFromRulesCc(rand.nextBoolean())
        .internalSkylarkFlagTestCanary(rand.nextBoolean())
        .maxComputationSteps(rand.nextLong())
        .recordRuleInstantiationCallstack(rand.nextBoolean())
        .build();
  }

  private static StarlarkSemanticsOptions parseOptions(String... args) throws Exception {
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(StarlarkSemanticsOptions.class)
            .allowResidue(false)
            .build();
    parser.parse(Arrays.asList(args));
    return parser.getOptions(StarlarkSemanticsOptions.class);
  }
}
