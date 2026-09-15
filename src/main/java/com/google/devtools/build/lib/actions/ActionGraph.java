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

import javax.annotation.Nullable;

/**
 * An action graph.
 *
 * <p>Provides lookups of generating actions for artifacts.
 */
public interface ActionGraph {

  /**
   * Returns the Action that, when executed, gives rise to this file.
   *
   * <p>If this Artifact is a source file, null is returned. (We don't try to return a "no-op
   * action" because that would require creating a new no-op Action for every source file, since
   * each Action knows its outputs, so sharing all the no-ops is not an option.)
   *
   * <p>It's also possible for derived Artifacts to have null generating Actions when these actions
   * are unknown.
   */
  @Nullable
  ActionAnalysisMetadata getGeneratingAction(Artifact artifact);
}
