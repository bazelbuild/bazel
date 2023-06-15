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
package com.google.devtools.build.lib.analysis.producers;

import static com.google.devtools.build.lib.analysis.AspectCollection.buildAspectKey;
import static com.google.devtools.build.lib.analysis.AspectResolutionHelpers.aspectMatchesConfiguredTarget;

import com.google.devtools.build.lib.analysis.AspectCollection;
import com.google.devtools.build.lib.analysis.AspectValue;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.DuplicateException;
import com.google.devtools.build.lib.analysis.configuredtargets.MergedConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** Computes {@link ConfiguredAspect}s and merges them into a prerequisite. */
final class ConfiguredAspectProducer implements StateMachine {
  interface ResultSink {
    void acceptConfiguredAspectMergedTarget(int outputIndex, ConfiguredTargetAndData mergedTarget);

    void acceptConfiguredAspectError(DuplicateException error);
  }

  // -------------------- Input --------------------
  private final AspectCollection aspects;
  private final ConfiguredTargetAndData prerequisite;

  // -------------------- Output --------------------
  @Nullable private final NestedSetBuilder<Package> transitivePackages;
  private final ResultSink sink;
  private final int outputIndex;

  // -------------------- Internal State --------------------
  private final HashMap<AspectDescriptor, AspectValue> aspectValues = new HashMap<>();

  ConfiguredAspectProducer(
      AspectCollection aspects,
      ConfiguredTargetAndData prerequisite,
      ResultSink sink,
      int outputIndex,
      @Nullable NestedSetBuilder<Package> transitivePackages) {
    this.aspects = aspects;
    this.prerequisite = prerequisite;
    this.sink = sink;
    this.outputIndex = outputIndex;
    this.transitivePackages = transitivePackages;
  }

  @Override
  public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
    ConfiguredTargetKey baseKey =
        ConfiguredTargetKey.fromConfiguredTarget(prerequisite.getConfiguredTarget());
    getAspectKeys(aspects, baseKey)
        .forEach(
            (aspectDescriptor, aspectKey) ->
                tasks.lookUp(aspectKey, v -> aspectValues.put(aspectDescriptor, (AspectValue) v)));
    return this::processResult;
  }

  private StateMachine processResult(Tasks tasks, ExtendedEventHandler listener) {
    var usedAspects = aspects.getUsedAspects();
    var configuredAspects = new ArrayList<ConfiguredAspect>(usedAspects.size());
    for (AspectCollection.AspectDeps depAspect : usedAspects) {
      var value = aspectValues.get(depAspect.getAspect());
      if (!aspectMatchesConfiguredTarget(prerequisite, value.getAspect())) {
        continue;
      }
      configuredAspects.add(value.getConfiguredAspect());
      if (transitivePackages != null) {
        transitivePackages.addTransitive(value.getTransitivePackages());
      }
    }
    try {
      sink.acceptConfiguredAspectMergedTarget(
          outputIndex,
          prerequisite.fromConfiguredTarget(
              MergedConfiguredTarget.of(prerequisite.getConfiguredTarget(), configuredAspects)));
    } catch (DuplicateException e) {
      sink.acceptConfiguredAspectError(e);
    }
    return DONE;
  }

  private static Map<AspectDescriptor, AspectKey> getAspectKeys(
      AspectCollection aspects, ConfiguredTargetKey baseKey) {
    var result = new HashMap<AspectDescriptor, AspectKey>();
    for (AspectCollection.AspectDeps aspectDeps : aspects.getUsedAspects()) {
      buildAspectKey(aspectDeps, result, baseKey);
    }
    return result;
  }
}
