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

import com.google.common.collect.ImmutableRangeMap;
import com.google.common.collect.Range;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;

/**
 * Formatter to write java.util.logging messages out in single-line format.
 *
 * <p>Log entries contain the date and time (in UTC), log level (as letter and numerical value),
 * source location, thread ID, message and, if applicable, a stack trace.
 */
public class SingleLineFormatter extends Formatter {

  /** Single-character codes based on {@link Level}s. */
  private static final ImmutableRangeMap<Integer, Character> CODES_BY_LEVEL =
      ImmutableRangeMap.<Integer, Character>builder()
          .put(Range.atMost(Level.FINE.intValue()), 'D')
          .put(Range.open(Level.FINE.intValue(), Level.WARNING.intValue()), 'I')
          .put(Range.closedOpen(Level.WARNING.intValue(), Level.SEVERE.intValue()), 'W')
          .put(Range.atLeast(Level.SEVERE.intValue()), 'X')
          .build();

  /** A thread safe, immutable formatter that can be used by all without contention. */
  private static final DateTimeFormatter DATE_TIME_FORMAT =
      DateTimeFormatter.ofPattern("yyMMdd HH:mm:ss.SSS").withZone(ZoneOffset.UTC);

  @Override
  public String format(LogRecord rec) {
    StringBuilder buf = new StringBuilder();

    // Timestamp
    buf.append(
            DATE_TIME_FORMAT.format(Instant.ofEpochMilli(rec.getMillis()).atZone(ZoneOffset.UTC)))
        .append(':');

    // One character code for level
    buf.append(CODES_BY_LEVEL.get(rec.getLevel().intValue()));

    // The stack trace, if any
    Throwable thrown = rec.getThrown();
    if (thrown != null) {
      buf.append('T');
    }

    buf.append(' ');

    // Information about the source of the exception
    buf.append(rec.getThreadID())
        .append(" [")
        .append(rec.getSourceClassName())
        .append('.')
        .append(rec.getSourceMethodName())
        .append("] ");

    // The actual message
    buf.append(formatMessage(rec)).append('\n');

    if (thrown != null) {
      StringWriter sw = new StringWriter();
      PrintWriter pw = new PrintWriter(sw);
      thrown.printStackTrace(pw);
      pw.flush();
      buf.append(sw.toString());
    }

    return buf.toString();
  }
}
