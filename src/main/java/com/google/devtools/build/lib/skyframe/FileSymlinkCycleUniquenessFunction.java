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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** A value builder that has the side effect of reporting a file symlink cycle. */
public class FileSymlinkCycleUniquenessFunction implements SkyFunction {

  @SuppressWarnings("unchecked")  // Cast from Object to ImmutableList<RootedPath>.
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) {
    StringBuilder cycleMessage = new StringBuilder("circular symlinks detected\n");
    cycleMessage.append("[start of symlink cycle]\n");
    for (RootedPath rootedPath : (ImmutableList<RootedPath>) skyKey.argument()) {
      cycleMessage.append(rootedPath.asPath() + "\n");
    }
    cycleMessage.append("[end of symlink cycle]");
    // The purpose of this value builder is the side effect of emitting an error message exactly
    // once per build per unique cycle.
    env.getListener().handle(Event.error(cycleMessage.toString()));
    return FileSymlinkCycleUniquenessValue.INSTANCE;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
