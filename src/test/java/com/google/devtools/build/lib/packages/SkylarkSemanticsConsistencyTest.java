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
 * <ul>
 *   <li>If a new option is added to {@code SkylarkSemantics} but not to {@code
 *       SkylarkSemanticsOptions}, or vice versa, then the programmer will either be unable to
 *       implement its behavior, or unable to test it from the command line and add user
 *       documentation. We hope that the programmer notices this on their own.
 *
 *   <li>If {@link SkylarkSemanticsOptions#toSkylarkSemantics} or {@link
 *       SkylarkSemanticsCodec#deserialize} is not updated to set all fields of {@code
 *       SkylarkSemantics}, then it will fail immediately because all fields of {@link
 *       SkylarkSemantics.Builder} are mandatory.
 *
 *   <li>To catch a copy/paste error where the wrong field's data is threaded through {@code
 *       toSkylarkSemantics()} or {@code deserialize(...)}, we repeatedly generate matching random
 *       instances of the input and expected output objects.
 *
 *   <li>The {@link #checkDefaultsMatch} test ensures that there is no divergence between the
 *       default values of the two classes.
 *
 *   <li>There is no test coverage for failing to update the non-generated webpage documentation.
 *       So don't forget that!
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
    SkylarkSemanticsCodec codec = new SkylarkSemanticsCodec();
    for (int i = 0; i < NUM_RANDOM_TRIALS; i++) {
      SkylarkSemantics semantics = buildRandomSemantics(new Random(i));
      SkylarkSemantics deserialized =
          TestUtils.fromBytes(codec, TestUtils.toBytes(codec, semantics));
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
        "--incompatible_bzl_disallow_load_after_statement=" + rand.nextBoolean(),
        "--incompatible_checked_arithmetic=" + rand.nextBoolean(),
        "--incompatible_comprehension_variables_do_not_leak=" + rand.nextBoolean(),
        "--incompatible_depset_is_not_iterable=" + rand.nextBoolean(),
        "--incompatible_dict_literal_has_no_duplicates=" + rand.nextBoolean(),
        "--incompatible_disallow_dict_plus=" + rand.nextBoolean(),
        "--incompatible_disallow_keyword_only_args=" + rand.nextBoolean(),
        "--incompatible_disallow_toplevel_if_statement=" + rand.nextBoolean(),
        "--incompatible_disallow_uncalled_set_constructor=" + rand.nextBoolean(),
        "--incompatible_list_plus_equals_inplace=" + rand.nextBoolean(),
        "--incompatible_load_argument_is_label=" + rand.nextBoolean(),
        "--incompatible_new_actions_api=" + rand.nextBoolean(),
        "--incompatible_show_all_print_messages=" + rand.nextBoolean(),
        "--incompatible_string_is_not_iterable=" + rand.nextBoolean(),
        "--internal_do_not_export_builtins=" + rand.nextBoolean(),
        "--internal_skylark_flag_test_canary=" + rand.nextBoolean());
  }

  /**
   * Constructs a {@link SkylarkSemantics} object with random fields. Must access {@code rand} using
   * the same sequence of operations (for the same fields) as {@link #buildRandomOptions}.
   */
  private static SkylarkSemantics buildRandomSemantics(Random rand) {
    return SkylarkSemantics.builder()
        // <== Add new options here in alphabetic order ==>
        .incompatibleBzlDisallowLoadAfterStatement(rand.nextBoolean())
        .incompatibleCheckedArithmetic(rand.nextBoolean())
        .incompatibleComprehensionVariablesDoNotLeak(rand.nextBoolean())
        .incompatibleDepsetIsNotIterable(rand.nextBoolean())
        .incompatibleDictLiteralHasNoDuplicates(rand.nextBoolean())
        .incompatibleDisallowDictPlus(rand.nextBoolean())
        .incompatibleDisallowKeywordOnlyArgs(rand.nextBoolean())
        .incompatibleDisallowToplevelIfStatement(rand.nextBoolean())
        .incompatibleDisallowUncalledSetConstructor(rand.nextBoolean())
        .incompatibleListPlusEqualsInplace(rand.nextBoolean())
        .incompatibleLoadArgumentIsLabel(rand.nextBoolean())
        .incompatibleNewActionsApi(rand.nextBoolean())
        .incompatibleShowAllPrintMessages(rand.nextBoolean())
        .incompatibleStringIsNotIterable(rand.nextBoolean())
        .internalDoNotExportBuiltins(rand.nextBoolean())
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
