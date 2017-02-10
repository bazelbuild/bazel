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

package com.google.devtools.build.benchmark;

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

import java.text.ParseException;
import java.util.Date;

/**
 * A converter class that convert an input string to a {@link DateFilter} object.
 */
public class DateFilterConverter implements Converter<DateFilter> {

  public DateFilterConverter() {
    super();
  }

  @Override
  public DateFilter convert(String input) throws OptionsParsingException {
    if (input.isEmpty()) {
      return null;
    }

    String[] parts = input.split("\\.\\.");
    if (parts.length != 2) {
      throw new OptionsParsingException("Error parsing time_between option: no '..' found.");
    }
    if (parts[0].isEmpty()) {
      throw new OptionsParsingException(
          "Error parsing time_between option: start date not found");
    }
    if (parts[1].isEmpty()) {
      throw new OptionsParsingException(
          "Error parsing time_between option: end date not found");
    }

    // TODO(yueg): support more date formats
    try {
      Date from = DateFilter.DATE_FORMAT.parse(parts[0]);
      Date to = DateFilter.DATE_FORMAT.parse(parts[1]);
      return new DateFilter(from, to);
    } catch (ParseException e) {
      throw new OptionsParsingException(
          "Error parsing datetime, format should be: yyyy-MM-ddTHH:mm:ss");
    }
  }

  @Override
  public String getTypeDescription() {
    return "A date filter in format: <start date(time)>..<end date(time)>";
  }

}