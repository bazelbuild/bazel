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
package com.google.devtools.build.lib.bazel;

import static com.google.common.truth.Truth.assertThat;

import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link BazelWorkspaceStatusModule.BazelWorkspaceStatusAction#parseWorkspaceStatus}. */
@RunWith(JUnit4.class)
public class BazelWorkspaceStatusModuleTest {

  private static Map<String, String> parse(String input) {
    return BazelWorkspaceStatusModule.BazelWorkspaceStatusAction.parseWorkspaceStatus(input);
  }

  @Test
  public void parseWorkspaceStatus_normalKeyValue() {
    assertThat(parse("KEY value\n")).containsExactly("KEY", "value");
  }

  @Test
  public void parseWorkspaceStatus_keyOnly_noSpace() {
    assertThat(parse("KEY_ONLY\n")).containsExactly("KEY_ONLY", "");
  }

  @Test
  public void parseWorkspaceStatus_keyWithTrailingSpace() {
    assertThat(parse("KEY_WITH_SPACE \n")).containsExactly("KEY_WITH_SPACE", "");
  }

  @Test
  public void parseWorkspaceStatus_keyValueAtEof_noTrailingNewline() {
    assertThat(parse("KEY value")).containsExactly("KEY", "value");
  }

  @Test
  public void parseWorkspaceStatus_keyOnlyAtEof_noTrailingNewline() {
    // Critical bug fix: input.trim() was stripping trailing space, making "KEY " look like "KEY"
    assertThat(parse("KEY_ONLY")).containsExactly("KEY_ONLY", "");
  }

  @Test
  public void parseWorkspaceStatus_keyWithSpaceAtEof_noTrailingNewline() {
    // Critical bug fix: "KEY " at EOF with no newline was being discarded
    assertThat(parse("KEY ")).containsExactly("KEY", "");
  }

  @Test
  public void parseWorkspaceStatus_multipleKeys() {
    assertThat(parse("STABLE_KEY stable_val\nVOLATILE_KEY volatile_val\n"))
        .containsExactly("STABLE_KEY", "stable_val", "VOLATILE_KEY", "volatile_val");
  }

  @Test
  public void parseWorkspaceStatus_mixedKeyOnlyAndKeyValue() {
    assertThat(parse("KEY_ONLY\nKEY value\n"))
        .containsExactly("KEY_ONLY", "", "KEY", "value");
  }

  @Test
  public void parseWorkspaceStatus_mixedKeyValueAndKeyOnly() {
    assertThat(parse("\n\nKEY value\nKEY_ONLY\n"))
        .containsExactly("KEY", "value", "KEY_ONLY", "");
  }

  @Test
  public void parseWorkspaceStatus_emptyLinesSkipped() {
    assertThat(parse("KEY value\n\nOTHER other\n"))
        .containsExactly("KEY", "value", "OTHER", "other");
  }

  @Test
  public void parseWorkspaceStatus_emptyInput() {
    assertThat(parse("")).isEmpty();
  }

  @Test
  public void parseWorkspaceStatus_valueWithSpaces() {
    assertThat(parse("KEY value with spaces\n"))
        .containsExactly("KEY", "value with spaces");
  }

  @Test
  public void parseWorkspaceStatus_valueTrimsTrailingWhitespace() {
    // Values are trimmed of trailing whitespace
    assertThat(parse("KEY value  \n")).containsExactly("KEY", "value");
  }
}
