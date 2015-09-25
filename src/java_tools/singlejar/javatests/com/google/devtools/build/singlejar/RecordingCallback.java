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

package com.google.devtools.build.singlejar;


import com.google.devtools.build.singlejar.ZipEntryFilter.CustomMergeStrategy;
import com.google.devtools.build.singlejar.ZipEntryFilter.StrategyCallback;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * A helper implementation of {@link StrategyCallback} that records callback
 * invocations as string.
 */
public final class RecordingCallback implements StrategyCallback {

  public final List<String> calls = new ArrayList<>();
  public final List<Date> dates = new ArrayList<>();

  @Override
  public void copy(Date date) {
    calls.add("copy");
    dates.add(date);
  }

  @Override
  public void rename(String filename, Date date) {
    calls.add("rename");
    dates.add(date);
  }

  @Override
  public void customMerge(Date date, CustomMergeStrategy strategy) {
    calls.add("customMerge");
    dates.add(date);
  }

  @Override
  public void skip() {
    calls.add("skip");
  }
}
