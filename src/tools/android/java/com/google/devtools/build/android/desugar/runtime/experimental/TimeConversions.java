// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.runtime;

import android.telephony.SubscriptionPlan;

/**
 * Conversions between built-in and desugared java.time primitives for calling built-in Android APIs
 * (see b/79121791).
 */
@SuppressWarnings("AndroidApiChecker")
public final class TimeConversions {
  private TimeConversions() {} // static methods only

  public static android.view.textclassifier.TextClassification.Request.Builder setReferenceTime(
      android.view.textclassifier.TextClassification.Request.Builder builder,
      j$.time.ZonedDateTime arg) {
    return builder.setReferenceTime(toZonedDateTime(arg));
  }

  /** The factory method for the {@link android.app.admin.FreezePeriod}. */
  @SuppressWarnings("MethodName") // synthetic method.
  public static android.app.admin.FreezePeriod create$FreezePeriod(
      j$.time.MonthDay jStart, j$.time.MonthDay jEnd) {
    java.time.MonthDay start = toMonthDay(jStart);
    java.time.MonthDay end = toMonthDay(jEnd);
    return new android.app.admin.FreezePeriod(start, end);
  }

  public static j$.time.MonthDay getStart(android.app.admin.FreezePeriod freezePeriod) {
    return fromMonthDay(freezePeriod.getStart());
  }

  public static j$.time.MonthDay getEnd(android.app.admin.FreezePeriod freezePeriod) {
    return fromMonthDay(freezePeriod.getEnd());
  }

  public static j$.time.ZonedDateTime getReferenceTime(
      android.view.textclassifier.TextClassification.Request request) {
    return fromZonedDateTime(request.getReferenceTime());
  }

  private static j$.time.MonthDay fromMonthDay(java.time.MonthDay monthDay) {
    return monthDay == null
        ? null
        : j$.time.MonthDay.of(monthDay.getMonthValue(), monthDay.getDayOfMonth());
  }

  private static java.time.MonthDay toMonthDay(j$.time.MonthDay monthDay) {
    return monthDay == null
        ? null
        : java.time.MonthDay.of(monthDay.getMonthValue(), monthDay.getDayOfMonth());
  }

  private static j$.time.ZonedDateTime fromZonedDateTime(java.time.ZonedDateTime dateTime) {
    if (dateTime == null) {
      return null;
    }
    return j$.time.ZonedDateTime.of(
        dateTime.getYear(),
        dateTime.getMonthValue(),
        dateTime.getDayOfMonth(),
        dateTime.getHour(),
        dateTime.getMinute(),
        dateTime.getSecond(),
        dateTime.getNano(),
        j$.time.ZoneId.of(dateTime.getZone().getId()));
  }

  private static java.time.ZonedDateTime toZonedDateTime(j$.time.ZonedDateTime dateTime) {
    if (dateTime == null) {
      return null;
    }
    return java.time.ZonedDateTime.of(
        dateTime.getYear(),
        dateTime.getMonthValue(),
        dateTime.getDayOfMonth(),
        dateTime.getHour(),
        dateTime.getMinute(),
        dateTime.getSecond(),
        dateTime.getNano(),
        java.time.ZoneId.of(dateTime.getZone().getId()));
  }

  public static SubscriptionPlan.Builder createNonrecurring(
      j$.time.ZonedDateTime start, j$.time.ZonedDateTime end) {
    return SubscriptionPlan.Builder.createNonrecurring(
        toZonedDateTime(start), toZonedDateTime(end));
  }

  public static SubscriptionPlan.Builder createRecurringMonthly(j$.time.ZonedDateTime start) {
    return SubscriptionPlan.Builder.createRecurringMonthly(toZonedDateTime(start));
  }
}
