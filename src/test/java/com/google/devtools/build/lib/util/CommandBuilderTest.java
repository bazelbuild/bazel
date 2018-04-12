// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import java.io.File;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for the {@link CommandBuilder} class.
 */
@RunWith(JUnit4.class)
public class CommandBuilderTest {

  private CommandBuilder builder() {
    return new CommandBuilder(new File("dummy-workdir"));
  }

  private void assertArgv(CommandBuilder builder, String... expected) {
    assertThat(builder.build().getCommandLineElements())
        .asList()
        .containsExactlyElementsIn(Arrays.asList(expected))
        .inOrder();
  }

  private void assertFailure(CommandBuilder builder, String expected) {
    try {
      builder.build();
      fail("Expected exception");
    } catch (Exception e) {
      assertThat(e).hasMessage(expected);
    }
  }

  @Test
  public void builderTest() {
    assertArgv(builder().addArg("abc"), "abc");
    assertArgv(builder().addArg("abc def"), "abc def");
    assertArgv(builder().addArgs("abc", "def"), "abc", "def");
    assertArgv(builder().addArgs(ImmutableList.of("abc", "def")), "abc", "def");
    assertArgv(builder().addArgs("/bin/sh", "-c", "abc"), "/bin/sh", "-c", "abc");
    assertArgv(builder().addArgs("/bin/sh", "-c"), "/bin/sh", "-c");
    assertArgv(builder().addArgs("/bin/bash", "-c"), "/bin/bash", "-c");
  }

  @Test
  public void failureScenarios() {
    assertFailure(builder(), "At least one argument is expected");
  }
}
