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

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

/** Test data for {@code KeepScanner} with references to java.* */
public class CollectionReferences {

  private final List<Date> dates;

  public CollectionReferences() {
    dates = new ArrayList<>(7);
    assert !(dates instanceof LinkedList);
  }

  @SuppressWarnings("unchecked")
  public void add(Date date) {
    List<Date> l = (AbstractList<Date>) Collection.class.cast(dates);
    l.add(date);
  }

  public Date first() {
    try {
      return dates.get(0);
    } catch (IndexOutOfBoundsException e) {
      return null;
    }
  }

  public long min() {
    long result = Long.MAX_VALUE; // compile-time constant, no ref
    for (Date d : dates) {
      if (d.getTime() < result) {
        result = d.getTime();
      }
    }
    return result;
  }

  public void expire(long before) {
    dates.removeIf(d -> d.getTime() < before);
  }

  static {
    System.out.println("Hello!");
  }
}
