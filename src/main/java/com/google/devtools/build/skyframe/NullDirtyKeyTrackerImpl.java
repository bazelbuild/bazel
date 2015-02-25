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

import com.google.common.collect.ImmutableSet;

import java.util.Set;

/**
 * Tracks nothing. Should be used by evaluators that don't do dirty node garbage collection.
 */
public class NullDirtyKeyTrackerImpl implements DirtyKeyTracker {

  @Override
  public void dirty(SkyKey skyKey) {
  }

  @Override
  public void notDirty(SkyKey skyKey) {
  }

  @Override
  public Set<SkyKey> getDirtyKeys() {
    return ImmutableSet.of();
  }
}
