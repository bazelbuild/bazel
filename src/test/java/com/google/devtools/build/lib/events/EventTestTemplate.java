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
package com.google.devtools.build.lib.events;

import com.google.devtools.build.lib.syntax.Location;
import org.junit.Before;

// Without 'public', tests fail in the guts of junit reflection.
// TODO(adonovan): copy this code into all subclasses.
// This is yet another terrible use of 'extends'.
public abstract class EventTestTemplate {

  protected Event event;
  protected String file;
  protected Location location;

  @Before
  public final void createLocations() throws Exception  {
    String message = "This is not an error message.";
    file = "/path/to/workspace/my/sample/path.txt";

    location = Location.fromFileLineColumn(file, 3, 4);

    event = Event.of(EventKind.WARNING, location, message);
  }
}
