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
package com.google.devtools.build.android.desugar.scan.testdata;

import java.util.ArrayList;
import java.util.Date;

/** Supplements {@link CollectionReferences} with additional and overlapping references to java.* */
public class OverlappingCollectionReferences {

  private final ArrayList<Date> dates;

  public OverlappingCollectionReferences() {
    dates = new ArrayList<>();
  }

  public void add(Date date) {
    dates.add(date);
  }

  public Date first() {
    try {
      return dates.get(0);
    } catch (IndexOutOfBoundsException e) {
      return null;
    }
  }

  public Date max() {
    long result = Long.MIN_VALUE; // compile-time constant, no ref
    for (Date d : dates) {
      if (d.getTime() > result) {
        result = d.getTime();
      }
    }
    return new Date(result);
  }
}
