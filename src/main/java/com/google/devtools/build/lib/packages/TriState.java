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

package com.google.devtools.build.lib.packages;

/**
 * Enum used to represent tri-state parameters in rule attributes (yes/no/auto).
 */
public enum TriState {
  YES,
  NO,
  AUTO;

  public int toInt() {
    return switch (this) {
      case YES -> 1;
      case NO -> 0;
      case AUTO -> -1;
    };
  }

  public static TriState fromInt(int n) {
    return switch (n) {
      case 1 -> YES;
      case 0 -> NO;
      case -1 -> AUTO;
      default -> throw new IllegalArgumentException("TriState must be -1, 0, or 1");
    };
  }
}
