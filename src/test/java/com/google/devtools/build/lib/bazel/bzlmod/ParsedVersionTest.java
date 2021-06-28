// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.bazel.bzlmod.ParsedVersion.ParseException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ParsedVersion}. */
@RunWith(JUnit4.class)
public class ParsedVersionTest {

  @Test
  public void testEmptyBeatsEverything() throws Exception {
    assertThat(ParsedVersion.parse("")).isGreaterThan(null);
    assertThat(ParsedVersion.parse("")).isGreaterThan(ParsedVersion.parse("1.0"));
    assertThat(ParsedVersion.parse("")).isGreaterThan(ParsedVersion.parse("1.0+build"));
    assertThat(ParsedVersion.parse("")).isGreaterThan(ParsedVersion.parse("1.0-pre"));
    assertThat(ParsedVersion.parse("")).isGreaterThan(ParsedVersion.parse("1.0-pre+build-kek.lol"));
  }

  @Test
  public void testEverythingBeatsNull() throws Exception {
    assertThat(ParsedVersion.parse("1.0")).isGreaterThan(null);
    assertThat(ParsedVersion.parse("1.0+build")).isGreaterThan(null);
    assertThat(ParsedVersion.parse("1.0-pre")).isGreaterThan(null);
    assertThat(ParsedVersion.parse("1.0-pre+build-kek")).isGreaterThan(null);
  }

  @Test
  public void testReleaseVersion() throws Exception {
    assertThat(ParsedVersion.parse("2.0")).isGreaterThan(ParsedVersion.parse("1.0"));
    assertThat(ParsedVersion.parse("2.0")).isGreaterThan(ParsedVersion.parse("1.9"));
    assertThat(ParsedVersion.parse("11.0")).isGreaterThan(ParsedVersion.parse("3.0"));
    assertThat(ParsedVersion.parse("1.0.1")).isGreaterThan(ParsedVersion.parse("1.0"));
    assertThat(ParsedVersion.parse("1.0.0")).isGreaterThan(ParsedVersion.parse("1.0"));
    assertThat(ParsedVersion.parse("1.0+build2"))
        .isEquivalentAccordingToCompareTo(ParsedVersion.parse("1.0+build3"));
    assertThat(ParsedVersion.parse("1.0")).isGreaterThan(ParsedVersion.parse("1.0-pre"));
    assertThat(ParsedVersion.parse("1.0"))
        .isEquivalentAccordingToCompareTo(ParsedVersion.parse("1.0+build-notpre"));
  }

  @Test
  public void testPrereleaseVersion() throws Exception {
    assertThat(ParsedVersion.parse("1.0-pre")).isGreaterThan(ParsedVersion.parse("1.0-are"));
    assertThat(ParsedVersion.parse("1.0-3")).isGreaterThan(ParsedVersion.parse("1.0-2"));
    assertThat(ParsedVersion.parse("1.0-pre")).isLessThan(ParsedVersion.parse("1.0-pre.foo"));
    assertThat(ParsedVersion.parse("1.0-pre.3")).isGreaterThan(ParsedVersion.parse("1.0-pre.2"));
    assertThat(ParsedVersion.parse("1.0-pre.10")).isGreaterThan(ParsedVersion.parse("1.0-pre.2"));
    assertThat(ParsedVersion.parse("1.0-pre.10a")).isLessThan(ParsedVersion.parse("1.0-pre.2a"));
    assertThat(ParsedVersion.parse("1.0-pre.99")).isLessThan(ParsedVersion.parse("1.0-pre.2a"));
  }

  @Test
  public void testParseException() throws Exception {
    assertThrows(ParseException.class, () -> ParsedVersion.parse("abc"));
    assertThrows(ParseException.class, () -> ParsedVersion.parse("1.0-pre?"));
    assertThrows(ParseException.class, () -> ParsedVersion.parse("1.0-pre///"));
    assertThrows(ParseException.class, () -> ParsedVersion.parse("1..0"));
    assertThrows(ParseException.class, () -> ParsedVersion.parse("1.0-pre..erp"));
  }
}
