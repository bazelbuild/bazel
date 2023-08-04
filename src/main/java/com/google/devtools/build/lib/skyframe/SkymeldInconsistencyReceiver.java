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
package com.google.devtools.build.lib.skyframe;


import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * The {@link GraphInconsistencyReceiver} that tolerates inconsistencies resulted in dropping
 * pre-execution nodes in Skymeld mode.
 */
public class SkymeldInconsistencyReceiver implements GraphInconsistencyReceiver {
  private static final ImmutableMap<SkyFunctionName, SkyFunctionName>
      SKYMELD_EXPECTED_MISSING_CHILDREN =
          ImmutableMap.of(
              SkyFunctions.ACTION_EXECUTION, SkyFunctions.GLOB,
              SkyFunctions.GLOB, SkyFunctions.GLOB);

  private final boolean heuristicallyDropNodes;

  public SkymeldInconsistencyReceiver(boolean heuristicallyDropNodes) {
    this.heuristicallyDropNodes = heuristicallyDropNodes;
  }

  @Override
  public void noteInconsistencyAndMaybeThrow(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    if (heuristicallyDropNodes
        && NodeDroppingInconsistencyReceiver.isExpectedInconsistency(
            key, otherKeys, inconsistency)) {
      // If `--heuristically_drop_nodes` is enabled, check whether the inconsistency is caused by
      // dropped state node. If so, tolerate the inconsistency and return.
      return;
    }

    if (!NodeDroppingInconsistencyReceiver.isExpectedInconsistency(
        key, otherKeys, inconsistency, SKYMELD_EXPECTED_MISSING_CHILDREN)) {
      // Instead of crashing, simply send a bug report here so we can evaluate whether this is an
      // actual bug or just something else to be added to the expected list.
      BugReport.logUnexpected(
          "Unexpected inconsistency: %s, %s, %s", key, otherKeys, inconsistency);
    }
  }
}
