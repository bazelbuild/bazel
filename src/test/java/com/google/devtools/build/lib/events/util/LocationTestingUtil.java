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
package com.google.devtools.build.lib.events.util;

import com.google.devtools.build.lib.events.Location;

import junit.framework.Assert;

/**
 * Static utility methods for testing Locations.
 */
public class LocationTestingUtil {

  private LocationTestingUtil() {
  }

  public static void assertEqualLocations(Location loc1, Location loc2) {
    Assert.assertEquals(loc1.getStartOffset(), loc2.getStartOffset());
    Assert.assertEquals(loc1.getStartLineAndColumn(), loc2.getStartLineAndColumn());
    Assert.assertEquals(loc1.getEndOffset(), loc2.getEndOffset());
    Assert.assertEquals(loc1.getEndLineAndColumn(), loc2.getEndLineAndColumn());
    Assert.assertEquals(loc1.getPath(), loc2.getPath());
  }
}
