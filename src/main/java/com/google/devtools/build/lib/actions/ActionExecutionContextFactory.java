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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import java.util.Collection;
import java.util.Map;

/**
 * Interface that provides an {@link ActionExecutionContext} on demand. Used to limit the surface
 * area available to callers that need to execute an action but do not need the full framework
 * normally provided.
 */
public interface ActionExecutionContextFactory {
  /**
   * Returns an action execution context. This involves creating a new FileOutErr, and it is the
   * caller's responsibility to close that, e.g. by calling {@link ActionExecutionContext#close}.
   */
  ActionExecutionContext getContext(ActionInputFileCache graphFileCache,
      MetadataHandler metadataHandler, Map<Artifact, Collection<Artifact>> expandedInputs);
}
