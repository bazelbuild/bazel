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
package com.google.devtools.build.skyframe;

import com.google.common.collect.Interner;
import com.google.common.collect.Interners;

/** Versioning scheme based on integers. */
final class IntVersion implements Version {
  private static final Interner<IntVersion> interner = Interners.newWeakInterner();

  private final long val;

  private IntVersion(long val) {
    this.val = val;
  }

  long getVal() {
    return val;
  }

  IntVersion next() {
    return of(val + 1);
  }

  static IntVersion of(long val) {
    return interner.intern(new IntVersion(val));
  }

  @Override
  public boolean atMost(Version other) {
    if (!(other instanceof IntVersion)) {
      return false;
    }
    return val <= ((IntVersion) other).val;
  }

  @Override
  public int hashCode() {
    return Long.valueOf(val).hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof IntVersion) {
      IntVersion other = (IntVersion) obj;
      return other.val == val;
    }
    return false;
  }

  @Override
  public String toString() {
    return "IntVersion: " + val;
  }
}
