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

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.UseDefault;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.ExpansionFunction;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.IsolatedOptionsData;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the Incompatible Changes system (--incompatible_* flags). These go in their own suite
 * because the options parser doesn't know the business logic for incompatible changes.
 */
@RunWith(JUnit4.class)
public class AllIncompatibleChangesExpansionTest {

  /** Dummy comment (linter suppression) */
  public static class ExampleOptions extends OptionsBase {
    @Option(
      name = "all",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      expansionFunction = AllIncompatibleChangesExpansion.class
    )
    public Void all;

    @Option(
      name = "X",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean x;

    @Option(
      name = "Y",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "true"
    )
    public boolean y;

    @Option(
      name = "incompatible_A",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "Migrate to A"
    )
    public boolean incompatibleA;

    @Option(
      name = "incompatible_B",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "Migrate to B"
    )
    public boolean incompatibleB;

    @Option(
        name = "incompatible_C",
        oldName = "experimental_C",
        metadataTags = {
            OptionMetadataTag.INCOMPATIBLE_CHANGE,
            OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "true",
        help = "Migrate to C"
    )
    public boolean incompatibleC;
  }

  /** Dummy comment (linter suppression) */
  public static class ExampleExpansionOptions extends OptionsBase {
    @Option(
      name = "incompatible_expX",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      expansion = {"--X"},
      help = "Start using X"
    )
    public Void incompatibleExpX;

    /** Dummy comment (linter suppression) */
    public static class NoYExpansion implements ExpansionFunction {
      @Override
      public ImmutableList<String> getExpansion(IsolatedOptionsData optionsData) {
        return ImmutableList.of("--noY");
      }
    }

    @Option(
      name = "incompatible_expY",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      expansionFunction = NoYExpansion.class,
      help = "Stop using Y"
    )
    public Void incompatibleExpY;
  }

  @Test
  public void noChangesSelected() throws OptionsParsingException {
    OptionsParser parser = OptionsParser.builder().optionsClasses(ExampleOptions.class).build();
    parser.parse("");
    ExampleOptions opts = parser.getOptions(ExampleOptions.class);
    assertThat(opts.x).isFalse();
    assertThat(opts.y).isTrue();
    assertThat(opts.incompatibleA).isFalse();
    assertThat(opts.incompatibleB).isFalse();
    assertThat(opts.incompatibleC).isTrue();
  }

  @Test
  public void allChangesSelected() throws OptionsParsingException {
    OptionsParser parser = OptionsParser.builder().optionsClasses(ExampleOptions.class).build();
    parser.parse("--all");
    ExampleOptions opts = parser.getOptions(ExampleOptions.class);
    assertThat(opts.x).isFalse();
    assertThat(opts.y).isTrue();
    assertThat(opts.incompatibleA).isTrue();
    assertThat(opts.incompatibleB).isTrue();
    assertThat(opts.incompatibleC).isTrue();
  }

  @Test
  public void rightmostOverrides() throws OptionsParsingException {
    // Check that all-expansion behaves just like any other expansion flag:
    // the rightmost setting of any individual option wins.
    OptionsParser parser = OptionsParser.builder().optionsClasses(ExampleOptions.class).build();
    parser.parse("--noincompatible_A", "--all", "--noincompatible_B");
    ExampleOptions opts = parser.getOptions(ExampleOptions.class);
    assertThat(opts.incompatibleA).isTrue();
    assertThat(opts.incompatibleB).isFalse();
  }

  @Test
  public void expansionOptions() throws OptionsParsingException {
    // Check that all-expansion behaves just like any other expansion flag:
    // the rightmost setting of any individual option wins.
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(ExampleOptions.class, ExampleExpansionOptions.class)
            .build();
    parser.parse("--all");
    ExampleOptions opts = parser.getOptions(ExampleOptions.class);
    assertThat(opts.x).isTrue();
    assertThat(opts.y).isFalse();
    assertThat(opts.incompatibleA).isTrue();
    assertThat(opts.incompatibleB).isTrue();
  }

  @Test
  public void invocationPolicy() throws OptionsParsingException {
    // Check that all-expansion behaves just like any other expansion flag and can be filtered
    // by invocation policy.
    InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
    invocationPolicyBuilder
        .addFlagPoliciesBuilder()
        .setFlagName("incompatible_A")
        .setUseDefault(UseDefault.getDefaultInstance());
    InvocationPolicy policy = invocationPolicyBuilder.build();
    InvocationPolicyEnforcer enforcer = new InvocationPolicyEnforcer(policy);

    OptionsParser parser = OptionsParser.builder().optionsClasses(ExampleOptions.class).build();
    parser.parse("--all");
    enforcer.enforce(parser);

    ExampleOptions opts = parser.getOptions(ExampleOptions.class);
    assertThat(opts.x).isFalse();
    assertThat(opts.y).isTrue();
    assertThat(opts.incompatibleA).isFalse(); // A should have been removed from the expansion.
    assertThat(opts.incompatibleB).isTrue(); // B, without a policy, should have been left alone.
  }

  /** Option with the right prefix, but the wrong metadata tag. */
  public static class IncompatibleChangeTagOption extends OptionsBase {
    @Option(
      name = "some_option_with_a_tag",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp"
    )
    public boolean opt;
  }

  @Test
  public void incompatibleChangeTagDoesNotTriggerAllIncompatibleChangesCheck() {
    try {
      OptionsParser.builder()
          .optionsClasses(ExampleOptions.class, IncompatibleChangeTagOption.class)
          .build();
    } catch (OptionsParser.ConstructionException e) {
      fail(
          "some_option_with_a_tag should not trigger the expansion, so there should be no checks "
              + "on it having the right prefix and metadata tags. Instead, the following exception "
              + "was thrown: "
              + e.getMessage());
    }
  }

  // There's no unit test to check that the expansion of --all is sorted. IsolatedOptionsData is not
  // exposed from OptionsParser, making it difficult to check, and it's not clear that exposing it
  // would be worth it.

  /**
   * Ensure that we get an {@link OptionsParser.ConstructionException} containing {@code message}
   * when the incompatible changes in the given {@link OptionsBase} subclass are validated.
   */
  // Because javadoc can't resolve inner classes.
  @SuppressWarnings("javadoc")
  private static void assertBadness(Class<? extends OptionsBase> optionsBaseClass, String message) {
    OptionsParser.ConstructionException e =
        assertThrows(
            "Should have failed with message \"" + message + "\"",
            OptionsParser.ConstructionException.class,
            () ->
                OptionsParser.builder()
                    .optionsClasses(ExampleOptions.class, optionsBaseClass)
                    .build());
    assertThat(e).hasMessageThat().contains(message);
  }

  /** Dummy comment (linter suppression) */
  public static class BadNameOptions extends OptionsBase {
    @Option(
      name = "badname",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp"
    )
    public boolean bad;
  }

  @Test
  public void badName() {
    assertBadness(
        BadNameOptions.class,
        "Incompatible change option '--badname' must have name "
            + "starting with \"incompatible_\"");
  }

  /** Option with the right prefix, but the wrong metadata tag. */
  public static class MissingTriggeredByTagOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp"
    )
    public boolean bad;
  }

  @Test
  public void badTag() {
    assertBadness(
        MissingTriggeredByTagOptions.class,
        "must have metadata tag OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES");
  }

  /** Option with the right prefix, but the wrong metadata tag. */
  public static class MissingIncompatibleTagOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      metadataTags = {OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES},
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp"
    )
    public boolean bad;
  }

  @Test
  public void otherBadTag() {
    assertBadness(
        MissingIncompatibleTagOptions.class,
        "must have metadata tag OptionMetadataTag.INCOMPATIBLE_CHANGE");
  }

  /** Dummy comment (linter suppression) */
  public static class BadTypeOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "0",
      help = "nohelp"
    )
    public int bad;
  }

  @Test
  public void badType() {
    assertBadness(BadTypeOptions.class, "must have boolean type");
  }

  /** Dummy comment (linter suppression) */
  public static class BadHelpOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean bad;
  }

  @Test
  public void badHelp() {
    assertBadness(BadHelpOptions.class, "must have a \"help\" string");
  }

  /** Dummy comment (linter suppression) */
  public static class BadAbbrevOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp",
      abbrev = 'x'
    )
    public boolean bad;
  }

  @Test
  public void badAbbrev() {
    assertBadness(BadAbbrevOptions.class, "must not use the abbrev field");
  }

  /** Dummy comment (linter suppression) */
  public static class BadValueHelpOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp",
      valueHelp = "x"
    )
    public boolean bad;
  }

  @Test
  public void badValueHelp() {
    assertBadness(BadValueHelpOptions.class, "must not use the valueHelp field");
  }

  /** Dummy comment (linter suppression) */
  public static class BadConverterOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp",
      converter = Converters.BooleanConverter.class
    )
    public boolean bad;
  }

  @Test
  public void badConverter() {
    assertBadness(BadConverterOptions.class, "must not use the converter field");
  }

  /** Dummy comment (linter suppression) */
  public static class BadAllowMultipleOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      help = "nohelp",
      allowMultiple = true
    )
    public List<String> bad;
  }

  @Test
  public void badAllowMutliple() {
    assertBadness(BadAllowMultipleOptions.class, "must not use the allowMultiple field");
  }

  /** Dummy comment (linter suppression) */
  public static class BadImplicitRequirementsOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp",
      implicitRequirements = "--x"
    )
    public boolean bad;
  }

  @Test
  public void badImplicitRequirements() {
    assertBadness(
        BadImplicitRequirementsOptions.class, "must not use the implicitRequirements field");
  }

  /** Dummy comment (linter suppression) */
  public static class BadOldNameOptions extends OptionsBase {
    @Option(
      name = "incompatible_bad",
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false",
      help = "nohelp",
      oldName = "x"
    )
    public boolean bad;
  }

  @Test
  public void badOldName() {
    assertBadness(BadOldNameOptions.class, "must not use the oldName field");
  }
}
