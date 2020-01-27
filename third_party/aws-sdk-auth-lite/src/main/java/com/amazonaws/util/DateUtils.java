/*
 * Copyright 2010-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Portions copyright 2006-2009 James Murty. Please see LICENSE.txt
 * for applicable license terms and NOTICE.txt for applicable notices.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
package com.amazonaws.util;

import com.amazonaws.annotation.ThreadSafe;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.ZoneId;
import java.util.Date;

/**
 * Utilities for parsing and formatting dates.
 */
@ThreadSafe
public class DateUtils {

    private static final ZoneId GMT = ZoneId.of("GMT");

    /** ISO 8601 format */
    protected static final DateTimeFormatter iso8601DateFormat =
        DateTimeFormatter.ISO_DATE_TIME.withZone(GMT);

    /** Alternate ISO 8601 format without fractional seconds */
    protected static final DateTimeFormatter alternateIso8601DateFormat =
        DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'Z'").withZone(GMT);

    /**
     * Parses the specified date string as an ISO 8601 date and returns the Date
     * object.
     *
     * @param dateString
     *            The date string to parse.
     *
     * @return The parsed Date object.
     */
    public static Date parseISO8601Date(String dateString) {
        return doParseISO8601Date(dateString);
    }

    static Date doParseISO8601Date(final String dateStringOrig) {
        String dateString = dateStringOrig;

        // For EC2 Spot Fleet.
        if (dateString.endsWith("+0000")) {
            dateString = dateString
                    .substring(0, dateString.length() - 5)
                    .concat("Z");
        }

        try {
          // Normal case: nothing special here
          final LocalDateTime ldt = LocalDateTime.parse(dateString, iso8601DateFormat);
          return Date.from(ldt.atZone(ZoneId.systemDefault()).toInstant());
        } catch (IllegalArgumentException e) {
            try {
                final LocalDateTime ldt = LocalDateTime.parse(dateString, alternateIso8601DateFormat);
                return Date.from(ldt.atZone(ZoneId.systemDefault()).toInstant());
                // If the first ISO 8601 parser didn't work, try the alternate
                // version which doesn't include fractional seconds
            } catch(Exception oops) {
                // no the alternative route doesn't work; let's bubble up the original exception
                throw e;
            }
        }
    }

}
