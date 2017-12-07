// Copyright 2017 The Bazel Authors. All rights reserved.
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
import static java.time.temporal.ChronoUnit.MILLIS;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SingleLineFormatterTest {

  private static final ZonedDateTime TIMESTAMP =
      ZonedDateTime.of(2017, 04, 01, 17, 03, 43, 0, ZoneOffset.UTC).plus(142, MILLIS);

  @Test
  public void testFormat() {
    LogRecord logRecord = createLogRecord(Level.SEVERE, TIMESTAMP);
    assertThat(new SingleLineFormatter().format(logRecord))
        .isEqualTo("170401 17:03:43.142:X 543 [SomeSourceClass.aSourceMethod] some message\n");
  }

  @Test
  public void testLevel() {
    LogRecord logRecord = createLogRecord(Level.WARNING, TIMESTAMP);
    String formatted = new SingleLineFormatter().format(logRecord);
    assertThat(formatted).contains("W");
    assertThat(formatted).doesNotContain("X");
  }

  @Test
  public void testTime() {
    LogRecord logRecord =
        createLogRecord(
            Level.SEVERE,
            ZonedDateTime.of(1999, 11, 30, 03, 04, 05, 0, ZoneOffset.UTC).plus(722, MILLIS));
    assertThat(new SingleLineFormatter().format(logRecord)).contains("991130 03:04:05.722");
  }

  @Test
  public void testStackTrace() {
    LogRecord logRecord = createLogRecord(
        Level.SEVERE, TIMESTAMP, new RuntimeException("something wrong"));
    assertThat(new SingleLineFormatter().format(logRecord))
        .startsWith(
            "170401 17:03:43.142:XT 543 [SomeSourceClass.aSourceMethod] some message\n"
            + "java.lang.RuntimeException: something wrong\n"
            + "\tat com.google.devtools.build.lib.util.SingleLineFormatterTest.testStackTrace");
  }

  private static LogRecord createLogRecord(Level level, ZonedDateTime dateTime) {
    return createLogRecord(level, dateTime, null);
  }

  private static LogRecord createLogRecord(
      Level level, ZonedDateTime dateTime, RuntimeException thrown) {
    LogRecord record = new LogRecord(level, "some message");
    record.setMillis(dateTime.toInstant().toEpochMilli());
    record.setSourceClassName("SomeSourceClass");
    record.setSourceMethodName("aSourceMethod");
    record.setThreadID(543);
    if (thrown != null) {
      record.setThrown(thrown);
    }
    return record;
  }
}
