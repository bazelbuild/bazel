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
package com.google.devtools.build.lib.events.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.events.Location;

/**
 * Static utility methods for testing Locations.
 */
public class LocationTestingUtil {

  private LocationTestingUtil() {
  }

  public static void assertEqualLocations(Location expected, Location actual) {
    assertThat(actual.getStartOffset()).isEqualTo(expected.getStartOffset());
    assertThat(actual.getStartLineAndColumn()).isEqualTo(expected.getStartLineAndColumn());
    assertThat(actual.getEndOffset()).isEqualTo(expected.getEndOffset());
    assertThat(actual.getEndLineAndColumn()).isEqualTo(expected.getEndLineAndColumn());
    assertThat(actual.getPath()).isEqualTo(expected.getPath());
  }
}
