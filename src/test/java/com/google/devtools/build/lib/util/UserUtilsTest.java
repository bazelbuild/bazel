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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link UserUtils}. */
@RunWith(JUnit4.class)
public final class UserUtilsTest {

  @Test
  public void sanitizeUserName_replacesSlashesAndBackslashes() {
    assertThat(UserUtils.sanitizeUserName("foo/bar\\baz")).isEqualTo("foo_bar_baz");
    assertThat(UserUtils.sanitizeUserName("DOMAIN\\user")).isEqualTo("DOMAIN_user");
    assertThat(UserUtils.sanitizeUserName("normaluser")).isEqualTo("normaluser");
    assertThat(UserUtils.sanitizeUserName(null)).isNull();
  }

  @Test
  public void getUserName_doesNotContainSlashOrBackslash() {
    String userName = UserUtils.getUserName();
    assertThat(userName).doesNotContain("\\");
    assertThat(userName).doesNotContain("/");
  }
}
