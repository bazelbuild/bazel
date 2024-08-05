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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.config.NativeAndStarlarkFlags;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link PlatformFlagsProducer}. */
@RunWith(JUnit4.class)
public class PlatformFlagsProducerTest extends ProducerTestCase {
  @Test
  public void nativeFlag() throws Exception {
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        platform(
            name = "basic",
            flags = [
                "--compilation_mode=dbg",
            ],
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    NativeAndStarlarkFlags result = fetch(platformLabel);

    assertThat(result).isNotNull();
    assertThat(result.nativeFlags()).contains("--compilation_mode=dbg");
  }

  private void createStarlarkFlag() throws Exception {
    scratch.file(
        "flag/def.bzl",
        """
        def _impl(ctx):
            return []

        basic_flag = rule(
            implementation = _impl,
            build_setting = config.string(flag = True),
        )
        """);

    scratch.file(
        "flag/BUILD",
        """
        load(":def.bzl", "basic_flag")

        basic_flag(
            name = "flag",
            build_setting_default = "from_default",
        )
        """);
  }

  @Test
  public void starlarkFlag() throws Exception {
    createStarlarkFlag();
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        platform(
            name = "basic",
            flags = [
                "--//flag=from_platform",
            ],
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    NativeAndStarlarkFlags result = fetch(platformLabel);

    assertThat(result).isNotNull();
    assertThat(result.starlarkFlags()).containsAtLeast("//flag:flag", "from_platform");
  }

  // Regression test for https://github.com/bazelbuild/bazel/issues/23147
  @Test
  public void starlarkFlag_resetToDefault() throws Exception {
    createStarlarkFlag();
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        platform(
            name = "basic",
            flags = [
                "--//flag=from_default",
            ],
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    NativeAndStarlarkFlags result = fetch(platformLabel);

    assertThat(result).isNotNull();
    assertThat(result.starlarkFlags()).containsAtLeast("//flag:flag", "from_default");
  }

  @Test
  public void starlarkFlag_invalid() throws Exception {
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        platform(
            name = "basic",
            flags = [
                "--//unknown/starlark:flag=fake",
            ],
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    assertThrows(OptionsParsingException.class, () -> fetch(platformLabel));
  }

  private NativeAndStarlarkFlags fetch(Label platformLabel)
      throws InvalidPlatformException, InterruptedException, OptionsParsingException {
    PlatformFlagsSink sink = new PlatformFlagsSink(platformLabel);
    PlatformFlagsProducer producer =
        new PlatformFlagsProducer(platformLabel, sink, StateMachine.DONE);
    var unused = executeProducer(producer);
    return sink.parsedFlags();
  }

  /** Receiver for platform info from {@link PlatformFlagsProducer}. */
  private static class PlatformFlagsSink implements PlatformFlagsProducer.ResultSink {
    private final Label expectedPlatform;
    @Nullable private NativeAndStarlarkFlags parsedFlags = null;
    @Nullable private InvalidPlatformException invalidPlatformException = null;
    @Nullable private OptionsParsingException optionParsingException = null;

    private PlatformFlagsSink(Label expectedPlatform) {
      this.expectedPlatform = expectedPlatform;
    }

    @Override
    public void acceptPlatformFlags(Label platform, NativeAndStarlarkFlags parsedFlags) {
      assertThat(platform).isEqualTo(this.expectedPlatform);
      this.parsedFlags = parsedFlags;
    }

    @Override
    public void acceptPlatformFlagsError(Label platform, InvalidPlatformException error) {
      assertThat(platform).isEqualTo(this.expectedPlatform);
      this.invalidPlatformException = error;
    }

    @Override
    public void acceptPlatformFlagsError(Label platform, OptionsParsingException error) {
      assertThat(platform).isEqualTo(this.expectedPlatform);
      this.optionParsingException = error;
    }

    NativeAndStarlarkFlags parsedFlags() throws InvalidPlatformException, OptionsParsingException {
      if (this.invalidPlatformException != null) {
        throw this.invalidPlatformException;
      }
      if (this.optionParsingException != null) {
        throw this.optionParsingException;
      }
      if (this.parsedFlags != null) {
        return parsedFlags;
      }
      throw new IllegalStateException("Value and exception not set");
    }
  }
}
