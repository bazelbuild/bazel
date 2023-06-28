// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.actions.cache.ActionCache;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;
import javax.annotation.Nullable;

/** Utility functions for {@link ActionCache}. */
public class ActionCacheUtils {
  private ActionCacheUtils() {}

  @Nullable
  public static Entry<String, ActionCache.Entry> getCacheEntryWithKey(
      ActionCache actionCache, Action action) {
    for (Artifact output : action.getOutputs()) {
      ActionCache.Entry entry = actionCache.get(output.getExecPathString());
      if (entry != null) {
        return new SimpleEntry<>(output.getExecPathString(), entry);
      }
    }
    return null;
  }

  /** Checks whether one of existing output paths is already used as a key. */
  @Nullable
  public static ActionCache.Entry getCacheEntry(ActionCache actionCache, Action action) {
    for (Artifact output : action.getOutputs()) {
      ActionCache.Entry entry = actionCache.get(output.getExecPathString());
      if (entry != null) {
        return entry;
      }
    }
    return null;
  }

  public static void removeCacheEntry(ActionCache actionCache, Action action) {
    for (Artifact output : action.getOutputs()) {
      actionCache.remove(output.getExecPathString());
    }
  }
}
