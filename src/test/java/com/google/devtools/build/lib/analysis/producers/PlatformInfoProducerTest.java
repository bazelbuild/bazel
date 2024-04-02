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

import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.state.StateMachine;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link PlatformInfoProducer}. */
@RunWith(JUnit4.class)
public class PlatformInfoProducerTest extends ProducerTestCase {
  @Test
  public void infoLookup() throws Exception {
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
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    PlatformInfo result = fetch(platformLabel);

    assertThat(result).isNotNull();
    assertThat(result.label()).isEqualTo(platformLabel);
  }

  @Test
  public void infoLookup_alias() throws Exception {
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
        )

        alias(
            name = "alias",
            actual = ":basic",
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:alias");
    PlatformInfo result = fetch(platformLabel);

    assertThat(result).isNotNull();
    assertThat(result.label()).isEqualTo(Label.parseCanonicalUnchecked("//lookup:basic"));
  }

  @Test
  public void infoLookup_error() throws Exception {
    scratch.overwriteFile(
        "lookup/BUILD",
        """
        filegroup(
            name = "basic",
        )
        """);

    Label platformLabel = Label.parseCanonicalUnchecked("//lookup:basic");
    assertThrows(InvalidPlatformException.class, () -> fetch(platformLabel));
  }

  private PlatformInfo fetch(Label platformLabel)
      throws InvalidPlatformException, InterruptedException {
    PlatformInfoSink sink = new PlatformInfoSink();
    PlatformInfoProducer producer =
        new PlatformInfoProducer(platformLabel, sink, StateMachine.DONE);
    assertThat(executeProducer(producer)).isTrue();
    return sink.platformInfo();
  }

  /** Receiver for platform info from {@link PlatformInfoProducer}. */
  private static class PlatformInfoSink implements PlatformInfoProducer.ResultSink {
    @Nullable private PlatformInfo platformInfo = null;
    @Nullable private InvalidPlatformException platformInfoError = null;

    @Override
    public void acceptPlatformInfo(PlatformInfo info) {
      this.platformInfo = info;
    }

    @Override
    public void acceptPlatformInfoError(InvalidPlatformException error) {
      this.platformInfoError = error;
    }

    PlatformInfo platformInfo() throws InvalidPlatformException {
      if (this.platformInfoError != null) {
        throw this.platformInfoError;
      }
      if (this.platformInfo != null) {
        return platformInfo;
      }
      throw new IllegalStateException("Value and exception not set");
    }
  }
}
