// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.packages.Target;
import java.util.Collection;

/**
 * A callback interface to notify the caller about specific events.
 * TODO(bazel-team): maybe we should use the EventBus instead?
 */
public interface LoadingCallback {
  /**
   * Called after the target patterns have been resolved to give the caller a chance to validate
   * the list before proceeding.
   */
  void notifyTargets(Collection<Target> targets) throws LoadingFailedException;
}