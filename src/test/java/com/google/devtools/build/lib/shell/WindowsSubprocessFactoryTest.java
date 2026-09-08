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

package com.google.devtools.build.lib.shell;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link WindowsSubprocessFactory}. */
@RunWith(JUnit4.class)
public class WindowsSubprocessFactoryTest {

  @Test
  public void testIsBatchFile() {
    assertThat(WindowsSubprocessFactory.isBatchFile("foo.bat")).isTrue();
    assertThat(WindowsSubprocessFactory.isBatchFile("foo.bat")).isTrue();
    assertThat(WindowsSubprocessFactory.isBatchFile("foo.cmd")).isTrue();
    assertThat(WindowsSubprocessFactory.isBatchFile("foo.btm")).isTrue();
    assertThat(WindowsSubprocessFactory.isBatchFile("FOO.BAT")).isTrue();
    assertThat(WindowsSubprocessFactory.isBatchFile("FOO.CMD")).isTrue();
    assertThat(WindowsSubprocessFactory.isBatchFile("FOO.BTM")).isTrue();
    assertThat(WindowsSubprocessFactory.isBatchFile("C:\\path\\to\\script.bat")).isTrue();

    assertThat(WindowsSubprocessFactory.isBatchFile("cmd.exe")).isFalse();
    assertThat(WindowsSubprocessFactory.isBatchFile("cmd")).isFalse();
    assertThat(WindowsSubprocessFactory.isBatchFile("4nt.exe")).isFalse();
    assertThat(WindowsSubprocessFactory.isBatchFile("foo.exe")).isFalse();
    assertThat(WindowsSubprocessFactory.isBatchFile("foo.sh")).isFalse();
    assertThat(WindowsSubprocessFactory.isBatchFile("bat")).isFalse();
    assertThat(WindowsSubprocessFactory.isBatchFile("foo.bat.exe")).isFalse();
    assertThat(WindowsSubprocessFactory.isBatchFile("mycmd.exe")).isFalse();
    assertThat(WindowsSubprocessFactory.isBatchFile("cmd.exe.bak")).isFalse();
  }

  @Test
  public void testContainsCmdMetaCharacters() {
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo bar")).isFalse();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("normal_arg")).isFalse();

    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo\nbar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo\rbar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo\"bar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo&bar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo|bar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo<bar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo>bar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo^bar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo%bar")).isTrue();
    assertThat(WindowsSubprocessFactory.containsCmdMetaCharacters("foo!bar")).isTrue();
  }
}
