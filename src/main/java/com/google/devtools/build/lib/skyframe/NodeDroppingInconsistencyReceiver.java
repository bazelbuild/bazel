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

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * {@link GraphInconsistencyReceiver} for evaluations operating on graphs when {@code
 * --experimental_heuristically_drop_nodes} flag is applied.
 *
 * <p>The expected inconsistency caused by heuristically dropping state nodes should be tolerated
 * while all other inconsistencies should result in throwing an exception.
 *
 * <p>{@code RewindableGraphInconsistencyReceiver} implements similar logic to handle heuristically
 * dropping state nodes.
 */
public class NodeDroppingInconsistencyReceiver implements GraphInconsistencyReceiver {

  @Override
  public void noteInconsistencyAndMaybeThrow(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    checkState(
        isExpectedInconsistency(key, otherKeys, inconsistency),
        "Unexpected inconsistency: %s, %s, %s",
        key,
        otherKeys,
        inconsistency);
  }

  /**
   * Checks whether the input inconsistency is an expected scenario caused by heuristically dropping
   * state nodes. See b/261019506 for background on this.
   */
  public static boolean isExpectedInconsistency(
      SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
    boolean isFileKey = key.functionName().equals(FileValue.FILE);
    boolean isDirectoryListingKey = key.functionName().equals(SkyFunctions.DIRECTORY_LISTING);
    if (!isFileKey && !isDirectoryListingKey) {
      return false;
    }
    if (inconsistency == Inconsistency.RESET_REQUESTED) {
      return otherKeys == null;
    }
    if (inconsistency == Inconsistency.ALREADY_DECLARED_CHILD_MISSING) {
      // For already declared child missing inconsistency, key is the parent while `otherKeys`
      // are the children (dependency nodes).
      SkyFunctionName expectedStateKeyType =
          isFileKey ? FileStateKey.FILE_STATE : SkyFunctions.DIRECTORY_LISTING_STATE;
      return otherKeys.stream().allMatch(SkyFunctionName.functionIs(expectedStateKeyType));
    }
    return false;
  }
}
