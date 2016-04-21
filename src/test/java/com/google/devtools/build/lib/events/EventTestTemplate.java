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

import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;

public abstract class EventTestTemplate {

  protected Event event;
  protected PathFragment path;
  protected Location location;
  protected Location locationNoPath;
  protected Location locationNoLineInfo;

  @Before
  public final void createLocations() throws Exception  {
    String message = "This is not an error message.";
    path = new PathFragment("/path/to/workspace/my/sample/path.txt");

    location = Location.fromPathAndStartColumn(path, 21, 31, new LineAndColumn(3, 4));

    event = Event.of(EventKind.WARNING, location, message);

    locationNoPath = Location.fromPathAndStartColumn(null, 21, 31, new LineAndColumn(3, 4));

    locationNoLineInfo = Location.fromFileAndOffsets(path, 21, 31);
  }
}
