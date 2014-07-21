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
package com.google.devtools.build.skyframe;

/**
 * Versioning scheme based on integers.
 */
public final class IntVersion implements Version {

  private final int val;

  public IntVersion(int val) {
    this.val = val;
  }

  public int getVal() {
    return val;
  }

  public IntVersion next() {
    return new IntVersion(val + 1);
  }

  @Override
  public int hashCode() {
    return val;
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
