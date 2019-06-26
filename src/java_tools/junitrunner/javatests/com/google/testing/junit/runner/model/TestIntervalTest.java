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
import static com.google.testing.junit.runner.model.TestInstantUtil.testInstant;

import com.google.testing.junit.runner.util.TestClock.TestInstant;
import java.time.Duration;
import java.time.Instant;
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
    Instant start = Instant.ofEpochMilli(123456);
    Instant end = Instant.ofEpochMilli(234567);
    TestInterval interval = new TestInterval(testInstant(start), testInstant(end));
    assertThat(interval.getStartMillis()).isEqualTo(123456);
    assertThat(interval.getEndMillis()).isEqualTo(234567);

    interval = new TestInterval(testInstant(start), testInstant(start));
    assertThat(interval.getStartMillis()).isEqualTo(123456);
    assertThat(interval.getEndMillis()).isEqualTo(123456);
  }

  @Test
  public void testCreationFailure() {
    thrown.expect(IllegalArgumentException.class);
    thrown.expectMessage("Start must be before end");
    new TestInterval(testInstant(Instant.ofEpochMilli(35)), testInstant(Instant.ofEpochMilli(23)));
  }

  @Test
  public void testToDuration() {
    assertThat(
            new TestInterval(
                    testInstant(Instant.ofEpochMilli(50)), testInstant(Instant.ofEpochMilli(150)))
                .toDurationMillis())
        .isEqualTo(100);
    assertThat(
            new TestInterval(
                    testInstant(Instant.ofEpochMilli(100)), testInstant(Instant.ofEpochMilli(100)))
                .toDurationMillis())
        .isEqualTo(0);
  }

  @Test
  public void testToDurationOnNonMonotonicWallTime() {
    Instant start = Instant.ofEpochMilli(123456);
    Instant end = Instant.ofEpochMilli(123456);
    Duration monotonicStart = Duration.ofMillis(50);
    Duration monotonicEnd = Duration.ofMillis(150);
    TestInterval interval =
        new TestInterval(
            new TestInstant(start, monotonicStart), new TestInstant(end, monotonicEnd));
    assertThat(interval.getStartMillis()).isEqualTo(123456);
    assertThat(interval.getEndMillis()).isEqualTo(123456);
    assertThat(interval.toDurationMillis()).isEqualTo(100);
  }

  @Test
  public void testDateFormat() {
    Date date = new Date(1471709734000L);
    TestInterval interval =
        new TestInterval(
            testInstant(date.toInstant()), testInstant(date.toInstant().plusMillis(100)));
    assertThat(interval.startInstantToString(TimeZone.getTimeZone("America/New_York")))
        .isEqualTo("2016-08-20T12:15:34.000-04:00");
    assertThat(interval.startInstantToString(TimeZone.getTimeZone("GMT")))
        .isEqualTo("2016-08-20T16:15:34.000Z");
  }
}
