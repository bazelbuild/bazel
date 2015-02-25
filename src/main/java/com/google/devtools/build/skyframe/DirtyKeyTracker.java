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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import java.util.Set;

/**
 * Interface for implementations that need to keep track of dirty SkyKeys.
 */
public interface DirtyKeyTracker {

  /**
   * Marks the {@code skyKey} as dirty.
   */
  @ThreadSafe
  void dirty(SkyKey skyKey);

  /**
   * Marks the {@code skyKey} as not dirty.
   */
  @ThreadSafe
  void notDirty(SkyKey skyKey);

  /**
   * Returns the set of keys k for which there was a call to dirty(k) but not a subsequent call
   * to notDirty(k).
   */
  @ThreadSafe
  Set<SkyKey> getDirtyKeys();
}
