// Copyright 2022 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.rules.genquery.GenQueryDirectPackageProviderFactory;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * {@link GraphInconsistencyReceiver} for evaluations operating on graphs when {@code
 * --heuristically_drop_nodes} flag is applied, or when some form of node dropping is done in
 * combination with skymeld mode.
 *
 * <p>The expected inconsistency should be tolerated while all other inconsistencies should result
 * in throwing an exception.
 *
 * <p>{@code RewindableGraphInconsistencyReceiver} implements similar logic to handle heuristically
 * dropping state nodes.
 */
public final class NodeDroppingInconsistencyReceiver implements GraphInconsistencyReceiver {

  private final boolean heuristicallyDropNodes;
  private final boolean skymeldInconsistenciesExpected;
  private static final ImmutableMap<SkyFunctionName, SkyFunctionName> EXPECTED_MISSING_CHILDREN =
      ImmutableMap.of(
          SkyFunctions.FILE, FileStateKey.FILE_STATE,
          SkyFunctions.DIRECTORY_LISTING, SkyFunctions.DIRECTORY_LISTING_STATE,
          SkyFunctions.CONFIGURED_TARGET, GenQueryDirectPackageProviderFactory.GENQUERY_SCOPE);

  // TODO: b/290998109#comment60 - After the GLOB nodes are replaced by GLOBS, the missing children
  // below might be unexpected.
  // These are only expected when Skymeld is enabled and we're dropping nodes.
  private static final ImmutableMap<SkyFunctionName, SkyFunctionName>
      SKYMELD_EXPECTED_MISSING_CHILDREN =
          ImmutableMap.of(SkyFunctions.ACTION_EXECUTION, SkyFunctions.GLOB);

  public NodeDroppingInconsistencyReceiver(
      boolean heuristicallyDropNodes, boolean skymeldInconsistenciesExpected) {
    this.heuristicallyDropNodes = heuristicallyDropNodes;
    this.skymeldInconsistenciesExpected = skymeldInconsistenciesExpected;
  }

  @Override
  public void noteInconsistencyAndMaybeThrow(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    if (heuristicallyDropNodes && isExpectedInconsistency(key, otherKeys, inconsistency)) {
      return;
    }
    if (skymeldInconsistenciesExpected
        && isExpectedInconsistencySkymeld(key, otherKeys, inconsistency)) {
      return;
    }

    throw new IllegalStateException(
        String.format("Unexpected inconsistency: %s, %s, %s", key, otherKeys, inconsistency));
  }

  /**
   * Checks whether the input inconsistency is an expected scenario caused by heuristically dropping
   * state nodes. See b/261019506 for background on this.
   */
  public static boolean isExpectedInconsistency(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    return isExpectedInternal(key, otherKeys, inconsistency, EXPECTED_MISSING_CHILDREN);
  }

  /**
   * Checks whether the input inconsistency is an expected scenario caused by skymeld + some form of
   * node dropping.
   */
  public static boolean isExpectedInconsistencySkymeld(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    return isExpectedInternal(key, otherKeys, inconsistency, SKYMELD_EXPECTED_MISSING_CHILDREN);
  }

  private static boolean isExpectedInternal(
      SkyKey key,
      @Nullable Collection<SkyKey> otherKeys,
      Inconsistency inconsistency,
      ImmutableMap<SkyFunctionName, SkyFunctionName> expectedMissingChildTypes) {
    SkyFunctionName expectedMissingChildType = expectedMissingChildTypes.get(key.functionName());
    if (expectedMissingChildType == null) {
      return false;
    }
    if (inconsistency == Inconsistency.RESET_REQUESTED) {
      return otherKeys == null;
    }
    if (inconsistency == Inconsistency.ALREADY_DECLARED_CHILD_MISSING
        || inconsistency == Inconsistency.BUILDING_PARENT_FOUND_UNDONE_CHILD) {
      // For already declared child missing inconsistency, key is the parent while `otherKeys`
      // are the children (dependency nodes).
      return otherKeys.stream().allMatch(SkyFunctionName.functionIs(expectedMissingChildType));
    }
    return false;
  }
}
