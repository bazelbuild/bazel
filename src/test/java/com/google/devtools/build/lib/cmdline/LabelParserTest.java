// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.cmdline.LabelParser.Parts.parse;
import static com.google.devtools.build.lib.cmdline.LabelParser.Parts.validateAndCreate;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LabelParser}. */
@RunWith(JUnit4.class)
public class LabelParserTest {

  /** Checks that the javadoc of {@link LabelParser.Parts#parse} is accurate. */
  @Test
  public void parserTable() throws Exception {
    assertThat(parse("foo/bar"))
        .isEqualTo(validateAndCreate(null, false, false, "", false, "foo/bar", "foo/bar"));
    assertThat(parse("...")).isEqualTo(validateAndCreate(null, false, false, "", true, "", "..."));
    assertThat(parse("...:all"))
        .isEqualTo(validateAndCreate(null, false, false, "", true, "all", "...:all"));
    assertThat(parse("foo/..."))
        .isEqualTo(validateAndCreate(null, false, false, "foo", true, "", "foo/..."));
    assertThat(parse("//foo/bar"))
        .isEqualTo(validateAndCreate(null, false, true, "foo/bar", false, "bar", "//foo/bar"));
    assertThat(parse("//foo/..."))
        .isEqualTo(validateAndCreate(null, false, true, "foo", true, "", "//foo/..."));
    assertThat(parse("//foo/...:all"))
        .isEqualTo(validateAndCreate(null, false, true, "foo", true, "all", "//foo/...:all"));
    assertThat(parse("//foo/all"))
        .isEqualTo(validateAndCreate(null, false, true, "foo/all", false, "all", "//foo/all"));
    assertThat(parse("@repo"))
        .isEqualTo(validateAndCreate("repo", false, true, "", false, "repo", "@repo"));
    assertThat(parse("@@repo"))
        .isEqualTo(validateAndCreate("repo", true, true, "", false, "repo", "@@repo"));
    assertThat(parse("@repo//foo/bar"))
        .isEqualTo(
            validateAndCreate("repo", false, true, "foo/bar", false, "bar", "@repo//foo/bar"));
    assertThat(parse("@@repo//foo/bar"))
        .isEqualTo(
            validateAndCreate("repo", true, true, "foo/bar", false, "bar", "@@repo//foo/bar"));
    assertThat(parse(":quux"))
        .isEqualTo(validateAndCreate(null, false, false, "", false, "quux", ":quux"));
    assertThat(parse(":qu:ux"))
        .isEqualTo(validateAndCreate(null, false, false, "", false, "qu:ux", ":qu:ux"));
    assertThat(parse("foo/bar:quux"))
        .isEqualTo(validateAndCreate(null, false, false, "foo/bar", false, "quux", "foo/bar:quux"));
    assertThat(parse("foo/bar:qu:ux"))
        .isEqualTo(validateAndCreate(null, false, false, "foo/bar", false, "qu:ux", "foo/bar:qu:ux"));
    assertThat(parse("//foo/bar:quux"))
        .isEqualTo(
            validateAndCreate(null, false, true, "foo/bar", false, "quux", "//foo/bar:quux"));
    assertThat(parse("@repo//foo/bar:quux"))
        .isEqualTo(
            validateAndCreate(
                "repo", false, true, "foo/bar", false, "quux", "@repo//foo/bar:quux"));
  }
}
