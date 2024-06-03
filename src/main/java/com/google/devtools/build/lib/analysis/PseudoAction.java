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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.Extension;
import com.google.protobuf.MessageLite;
import java.util.Collection;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * An action that is inserted into the build graph only to provide info
 * about rules to extra_actions.
 */
public class PseudoAction<InfoType extends MessageLite> extends AbstractAction {
  @VisibleForSerialization protected final UUID uuid;
  private final String mnemonic;

  @VisibleForSerialization protected final Extension<ExtraActionInfo, InfoType> infoExtension;

  private final InfoType info;

  public PseudoAction(
      UUID uuid,
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      Collection<Artifact> outputs,
      String mnemonic,
      Extension<ExtraActionInfo, InfoType> infoExtension,
      InfoType info) {
    super(owner, inputs, outputs);
    this.uuid = uuid;
    this.mnemonic = mnemonic;
    this.infoExtension = infoExtension;
    this.info = info;
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    String message = mnemonic + "ExtraAction should not be executed.";
    DetailedExitCode detailedCode =
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setExecution(
                    Execution.newBuilder().setCode(Code.PSEUDO_ACTION_EXECUTION_PROHIBITED))
                .build());
    throw new ActionExecutionException(message, this, false, detailedCode);
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addUUID(uuid);
    fp.addBytes(getInfo().toByteArray());
  }

  protected InfoType getInfo() {
    return this.info;
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext)
      throws CommandLineExpansionException, InterruptedException {
    return super.getExtraActionInfo(actionKeyContext).setExtension(infoExtension, getInfo());
  }

  public static Artifact getDummyOutput(RuleContext ruleContext) {
    return ruleContext.getPackageRelativeArtifact(
        ruleContext.getLabel().getName() + ".extra_action_dummy",
        ruleContext.getGenfilesDirectory());
  }
}
