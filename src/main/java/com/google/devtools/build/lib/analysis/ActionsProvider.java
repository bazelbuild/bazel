// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider;
import java.util.HashMap;
import java.util.Map;

/**
 * This provides a view over the actions that were created during the analysis of a rule
 * (not including actions for its transitive dependencies).
 */
public final class ActionsProvider {

  /** The Actions provider type itself. */
  public static final NativeProvider<Info> SKYLARK_CONSTRUCTOR =
      new NativeProvider<Info>(Info.class, "Actions") {};

  /** Factory method for creating instances of the Actions provider. */
  public static Info create(Iterable<ActionAnalysisMetadata> actions) {
    Map<Artifact, ActionAnalysisMetadata> map = new HashMap<>();
    for (ActionAnalysisMetadata action : actions) {
      for (Artifact artifact : action.getOutputs()) {
        // In the case that two actions generated the same artifact, the first wins. They
        // ought to be equal anyway.
        if (!map.containsKey(artifact)) {
          map.put(artifact, action);
        }
      }
    }
    ImmutableMap<String, Object> fields = ImmutableMap.<String, Object>of("by_file", map);
    return new Info(SKYLARK_CONSTRUCTOR, fields);
  }
}
