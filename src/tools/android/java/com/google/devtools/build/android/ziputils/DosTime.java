// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.ziputils;

import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;

/**
 * This class supports conversion from a {@code java.util.Date} object, to a
 * 4 bytes DOS date and time representation.
 */
public final class DosTime {

  /** DOS representation of DOS epoch (midnight, jan 1, 1980) */
  public static final DosTime EPOCH;
  /** {@code java.util.Date} for DOS epoch */
  public static final Date DOS_EPOCH;
  private static final Calendar calendar;

  /**
   * DOS representation of date passed to constructor.
   * Time is lower the 16 bit, date the upper 16 bit.
   */
  public final int time;

  /**
   * Creates a DOS representation of the given date.
   * @param date date to represent in DOS format.
   */
  public DosTime(Date date) {
    this.time = dateToDosTime(date);
  }

  static {
    calendar = new GregorianCalendar(1980, 0, 1, 0, 0, 0);
    DOS_EPOCH = calendar.getTime();
    EPOCH = new DosTime(DOS_EPOCH);
  }

  private static synchronized int dateToDosTime(Date date) {
    calendar.setTime(date);
    int year = calendar.get(Calendar.YEAR);
    if (year < 1980) {
      throw new IllegalArgumentException("date must be in or after 1980");
    }
    if (year > 2107) {
      throw new IllegalArgumentException("date must be before 2107");
    }
    int month = calendar.get(Calendar.MONTH) + 1;
    int day = calendar.get(Calendar.DAY_OF_MONTH);
    int hour = calendar.get(Calendar.HOUR_OF_DAY);
    int minute = calendar.get(Calendar.MINUTE);
    int second = calendar.get(Calendar.SECOND);
    return ((year - 1980) << 25) | (month << 21) | (day << 16)
        | (hour << 11) | (minute << 5) | (second >> 1);
  }
}
