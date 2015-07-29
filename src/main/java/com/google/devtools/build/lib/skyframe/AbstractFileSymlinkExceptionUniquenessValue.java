// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A value for ensuring that a file symlink error is reported exactly once. This is achieved by
 * forcing the same value key for two logically equivalent errors and letting Skyframe do its
 * magic.
 */
class AbstractFileSymlinkExceptionUniquenessValue implements SkyValue {
  protected static SkyKey key(SkyFunctionName functionName, ImmutableList<RootedPath> chain) {
    Preconditions.checkState(!chain.isEmpty());
    return new SkyKey(functionName, canonicalize(chain));
  }

  private static ImmutableList<RootedPath> canonicalize(ImmutableList<RootedPath> cycle) {
    int minPos = 0;
    String minString = cycle.get(0).toString();
    for (int i = 1; i < cycle.size(); i++) {
      String candidateString = cycle.get(i).toString();
      if (candidateString.compareTo(minString) < 0) {
        minPos = i;
        minString = candidateString;
      }
    }
    ImmutableList.Builder<RootedPath> builder = ImmutableList.builder();
    for (int i = 0; i < cycle.size(); i++) {
      int pos = (minPos + i) % cycle.size();
      builder.add(cycle.get(pos));
    }
    return builder.build();
  }
}

