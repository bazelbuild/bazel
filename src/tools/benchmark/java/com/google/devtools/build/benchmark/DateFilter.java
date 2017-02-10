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

import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Contains start date and end date.
 */
class DateFilter {

  static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");

  private final Date from;
  private final Date to;

  public DateFilter(Date from, Date to) {
    this.from = from;
    this.to = to;
  }

  public String getFromString() {
    return DATE_FORMAT.format(from);
  }

  public String getToString() {
    return DATE_FORMAT.format(to);
  }

}
