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
package com.google.devtools.build.lib.events;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A super simple little test for the {@link Event} class.
 */
@RunWith(JUnit4.class)
public class EventTest extends EventTestTemplate {

  @Test
  public void eventRetainsEventKind() {
    assertEquals(EventKind.WARNING, event.getKind());
  }

  @Test
  public void eventRetainsMessage() {
    assertEquals("This is not an error message.", event.getMessage());
  }

  @Test
  public void eventRetainsLocation() {
    assertEquals(21, event.getLocation().getStartOffset());
    assertEquals(31, event.getLocation().getEndOffset());
  }

}
