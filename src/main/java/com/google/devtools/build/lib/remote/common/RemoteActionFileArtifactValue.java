// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.common;

import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import java.util.Arrays;
import java.util.Objects;

/**
 * A {@link RemoteFileArtifactValue} with additional data only available when using Remote Execution
 * API (e.g. {@code isExecutable}).
 */
public class RemoteActionFileArtifactValue extends RemoteFileArtifactValue {

  private final boolean isExecutable;

  public RemoteActionFileArtifactValue(
      byte[] digest, long size, int locationIndex, String actionId, boolean isExecutable) {
    super(digest, size, locationIndex, actionId);
    this.isExecutable = isExecutable;
  }

  public boolean isExecutable() {
    return isExecutable;
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof RemoteActionFileArtifactValue)) {
      return false;
    }

    RemoteActionFileArtifactValue that = (RemoteActionFileArtifactValue) o;
    return Arrays.equals(getDigest(), that.getDigest())
        && getSize() == that.getSize()
        && getLocationIndex() == that.getLocationIndex()
        && Objects.equals(getActionId(), that.getActionId())
        && isExecutable == that.isExecutable
        && dataIsShareable() == that.dataIsShareable();
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        Arrays.hashCode(getDigest()),
        getSize(),
        getLocationIndex(),
        getActionId(),
        isExecutable,
        dataIsShareable());
  }
}
