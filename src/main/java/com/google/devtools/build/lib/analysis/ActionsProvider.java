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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.SkylarkInfo;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skylarkbuildapi.ActionsInfoProviderApi;
import com.google.devtools.build.lib.syntax.Starlark;
import java.util.HashMap;
import java.util.Map;

/**
 * This provides a view over the actions that were created during the analysis of a rule
 * (not including actions for its transitive dependencies).
 */
public final class ActionsProvider extends BuiltinProvider<StructImpl>
    implements ActionsInfoProviderApi {

  /** The ActionsProvider singleton instance. */
  public static final ActionsProvider INSTANCE = new ActionsProvider();

  public ActionsProvider() {
    super("Actions", StructImpl.class);
  }

  /** Factory method for creating instances of the Actions provider. */
  public static StructImpl create(Iterable<ActionAnalysisMetadata> actions) {
    Map<Artifact, ActionAnalysisMetadata> map = new HashMap<>();
    for (ActionAnalysisMetadata action : actions) {
      for (Artifact artifact : action.getOutputs()) {
        // In the case that two actions generated the same artifact, the first wins. They
        // ought to be equal anyway.
        map.putIfAbsent(artifact, action);
      }
    }
    ImmutableMap<String, Object> fields =
        ImmutableMap.<String, Object>of("by_file", Starlark.fromJava(map, /*mutability=*/ null));
    return SkylarkInfo.create(INSTANCE, fields, Location.BUILTIN);
  }
}
