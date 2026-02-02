// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.PlatformValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests of {@link PlatformProducer}.
 *
 * <p>Implicitly provides test coverage for {@link
 * com.google.devtools.build.lib.analysis.platform.PlatformFunction}.
 */
@RunWith(JUnit4.class)
public final class PlatformProducerTest extends ProducerTestCase {

  @Test
  public void basicLookup() throws Exception {
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        constraint_setting(name = "setting1")

        constraint_value(
            name = "value1",
            constraint_setting = ":setting1",
        )

        platform(
            name = "basic",
            constraint_values = [":value1"],
            flags = ["--cpu=fast"],
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    PlatformValue result = fetch(platformLabel, /* flagAliasMappings= */ ImmutableMap.of());

    assertThat(result).isNotNull();
    assertThat(result.platformInfo().label()).isEqualTo(platformLabel);
    assertThat(result.parsedFlags().get().parsingResult().canonicalize())
        .containsExactly("--cpu=fast");
  }

  @Test
  public void alias() throws Exception {
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        constraint_setting(name = "setting1")

        constraint_value(
            name = "value1",
            constraint_setting = ":setting1",
        )

        platform(
            name = "basic",
            constraint_values = [":value1"],
            flags = ["--cpu=fast"],
        )

        alias(
            name = "alias",
            actual = ":basic",
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:alias");
    PlatformValue result = fetch(platformLabel, /* flagAliasMappings= */ ImmutableMap.of());

    assertThat(result).isNotNull();
    assertThat(result.platformInfo().label())
        .isEqualTo(Label.parseCanonicalUnchecked("//lookup:basic"));
    assertThat(result.parsedFlags().get().parsingResult().canonicalize())
        .containsExactly("--cpu=fast");
  }

  @Test
  public void invalidPlatformError() throws Exception {
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        filegroup(
            name = "basic",
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    assertThrows(
        InvalidPlatformException.class,
        () -> fetch(platformLabel, /* flagAliasMappings= */ ImmutableMap.of()));
  }

  @Test
  public void optionsParsingError() throws Exception {
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        platform(
            name = "basic",
            flags = ["--//starlark:flag=does_not_exist"],
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    assertThrows(
        OptionsParsingException.class,
        () -> fetch(platformLabel, /* flagAliasMappings= */ ImmutableMap.of()));
  }

  @Test
  public void flagAliasUsesCanonicalFlag() throws Exception {
    scratch.overwriteFile(
        "starlark/flags.bzl",
        """
        string_flag = rule(implementation = lambda ctx: [], build_setting = config.string(flag = True))
        bool_flag = rule(implementation = lambda ctx: [], build_setting = config.bool(flag = True))
        """);
    scratch.overwriteFile(
        "starlark/BUILD",
        """
        load("//starlark:flags.bzl", "string_flag", "bool_flag")
        string_flag(
            name = "actual1",
            build_setting_default = "value1",
        )
        bool_flag(
            name = "actual2",
            build_setting_default = False,
        )
        bool_flag(
            name = "actual3",
            build_setting_default = True,
        )

        """);
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        constraint_setting(name = "setting1")
        constraint_value(
            name = "value1",
            constraint_setting = ":setting1",
        )
        platform(
            name = "basic",
            constraint_values = [":value1"],
            flags = [
                "--aliasname1=fast",
                "--aliasname2",
                "--noaliasname3",
                "--compilation_mode=opt",
            ],
        )
        """);

    PlatformValue result =
        fetch(
            Label.parseCanonicalUnchecked("//lookup:basic"),
            /* flagAliasMappings= */ ImmutableMap.of(
                "aliasname1", Label.parseCanonicalUnchecked("//starlark:actual1"),
                "aliasname2", Label.parseCanonicalUnchecked("//starlark:actual2"),
                "aliasname3", Label.parseCanonicalUnchecked("//starlark:actual3")));

    // Native flags:
    assertThat(result.parsedFlags().get().parsingResult().canonicalize())
        .containsExactly("--compilation_mode=opt");
    // Starlark flags:
    assertThat(result.parsedFlags().get().parsingResult().getStarlarkOptions())
        .containsExactly(
            "//starlark:actual1", "fast", "//starlark:actual2", true, "//starlark:actual3", false);
  }

  private PlatformValue fetch(
      Label platformLabel, ImmutableMap<String, Label> flagAliasMappings)
      throws InvalidPlatformException, OptionsParsingException, InterruptedException {
    PlatformInfoSink sink = new PlatformInfoSink();
    PlatformProducer producer =
        new PlatformProducer(platformLabel, flagAliasMappings, sink, StateMachine.DONE);
    boolean success = executeProducer(producer);
    if (sink.platformValue != null) {
      assertThat(success).isTrue();
      return sink.platformValue;
    } else {
      assertThat(success).isFalse(); // Error comes from a Skyframe dep.
      if (sink.platformInfoError != null) {
        throw sink.platformInfoError;
      } else {
        throw sink.optionsParsingError;
      }
    }
  }

  /** Receiver for platform info from {@link PlatformProducer}. */
  private static class PlatformInfoSink implements PlatformProducer.ResultSink {
    @Nullable private PlatformValue platformValue = null;
    @Nullable private InvalidPlatformException platformInfoError = null;
    @Nullable private OptionsParsingException optionsParsingError = null;

    @Override
    public void acceptPlatformValue(PlatformValue value) {
      this.platformValue = value;
    }

    @Override
    public void acceptPlatformInfoError(InvalidPlatformException error) {
      this.platformInfoError = error;
    }

    @Override
    public void acceptOptionsParsingError(OptionsParsingException error) {
      this.optionsParsingError = error;
    }
  }
}
