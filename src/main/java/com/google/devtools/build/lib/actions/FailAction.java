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

import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailAction.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import javax.annotation.Nullable;

/**
 * FailAction is an Action that always fails to execute. (Used as scaffolding for rules we haven't
 * yet implemented. Also useful for testing.)
 */
@Immutable
public final class FailAction extends AbstractAction {

  private static final String GUID = "626cb78a-810f-4af3-979c-ee194955f04c";

  private final String errorMessage;

  public FailAction(ActionOwner owner, Iterable<Artifact> outputs, String errorMessage) {
    super(owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), outputs);
    this.errorMessage = errorMessage;
  }

  @Override
  public Artifact getPrimaryInput() {
    return null;
  }

  public String getErrorMessage() {
    return errorMessage;
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    throw new ActionExecutionException(
        errorMessage,
        this,
        false,
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(
                    "FailAction intentional failure: "
                        + errorMessage
                        + " caused by "
                        + getOwner().getLabel())
                .setFailAction(
                    FailureDetails.FailAction.newBuilder().setCode(Code.INTENTIONAL_FAILURE))
                .build()));
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addString(GUID);
    // Should never be cached, but just be safe.
    fp.addString(getErrorMessage());
  }

  @Override
  protected String getRawProgressMessage() {
    return "Reporting failed target "
        + getOwner().getLabel()
        + " located at "
        + getOwner().getLocation();
  }

  @Override
  public String getMnemonic() {
    return "Fail";
  }
}
