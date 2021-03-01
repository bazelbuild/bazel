// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.cmdline.Label;

class RankAndLabel implements Comparable<RankAndLabel> {
  private final int rank;
  private final Label label;

  RankAndLabel(int rank, Label label) {
    this.rank = rank;
    this.label = label;
  }

  int getRank() {
    return rank;
  }

  @Override
  public int compareTo(RankAndLabel o) {
    if (this.rank != o.rank) {
      return this.rank - o.rank;
    }
    return this.label.compareTo(o.label);
  }

  @Override
  public String toString() {
    return rank + " " + label.getCanonicalForm();
  }
}