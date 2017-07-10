// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.model;

import static com.google.common.truth.Truth.assertThat;

import java.util.Date;
import java.util.TimeZone;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TestIntervalTest {
  @Rule public ExpectedException thrown = ExpectedException.none();

  @Test
  public void testCreation() {
    TestInterval interval = new TestInterval(123456, 234567);
    assertThat(interval.getStartMillis()).isEqualTo(123456);
    assertThat(interval.getEndMillis()).isEqualTo(234567);

    interval = new TestInterval(123456, 123456);
    assertThat(interval.getStartMillis()).isEqualTo(123456);
    assertThat(interval.getEndMillis()).isEqualTo(123456);
  }

  @Test
  public void testCreationFailure() {
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("Start must be before end");
    new TestInterval(35, 23);
  }

  @Test
  public void testToDuration() {
    assertThat(new TestInterval(50, 150).toDurationMillis()).isEqualTo(100);
    assertThat(new TestInterval(100, 100).toDurationMillis()).isEqualTo(0);
  }

  @Test
  public void testDateFormat() {
    Date date = new Date(1471709734000L);
    TestInterval interval = new TestInterval(date.getTime(), date.getTime() + 100);
    assertThat(interval.startInstantToString(TimeZone.getTimeZone("America/New_York")))
        .isEqualTo("2016-08-20T12:15:34.000-04:00");
    assertThat(interval.startInstantToString(TimeZone.getTimeZone("GMT")))
        .isEqualTo("2016-08-20T16:15:34.000Z");
  }
}
