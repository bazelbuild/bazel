// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.Collection;

/** Context to be informed of top-level outputs and their runfiles. */
public interface ImportantOutputHandler extends ActionContext {

  /**
   * Informs this handler that top-level outputs or their runfiles have been built.
   *
   * <p>The handler may verify that remotely stored outputs are still available. Returns a map from
   * digest to output for any artifacts that need to be regenerated via action rewinding.
   *
   * @throws ImportantOutputException for an issue processing the outputs, not including lost
   *     outputs which are reported in the returned map
   */
  ImmutableMap<String, ActionInput> processAndGetLostArtifacts(
      Collection<? extends ActionInput> outputs, InputMetadataProvider metadataProvider)
      throws ImportantOutputException, InterruptedException;

  ImportantOutputHandler NO_OP = (outputs, metadataProvider) -> ImmutableMap.of();

  /** Represents an exception encountered during {@link #processAndGetLostArtifacts}. */
  final class ImportantOutputException extends Exception implements DetailedException {
    private final FailureDetail failureDetail;

    public ImportantOutputException(Throwable cause, FailureDetail failureDetail) {
      super(failureDetail.getMessage(), cause);
      this.failureDetail = failureDetail;
    }

    public FailureDetail getFailureDetail() {
      return failureDetail;
    }

    @Override
    public DetailedExitCode getDetailedExitCode() {
      return DetailedExitCode.of(failureDetail);
    }
  }
}
