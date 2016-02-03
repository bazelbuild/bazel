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
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.GeneratedMessage.GeneratedExtension;
import com.google.protobuf.MessageLite;

import java.util.Collection;
import java.util.UUID;

/**
 * An action that is inserted into the build graph only to provide info
 * about rules to extra_actions.
 */
public class PseudoAction<InfoType extends MessageLite> extends AbstractAction {

  private final UUID uuid;
  private final String mnemonic;
  private final GeneratedExtension<ExtraActionInfo, InfoType> infoExtension;
  private final InfoType info;

  public PseudoAction(UUID uuid, ActionOwner owner,
      NestedSet<Artifact> inputs, Collection<Artifact> outputs,
      String mnemonic,
      GeneratedExtension<ExtraActionInfo, InfoType> infoExtension, InfoType info) {
    super(owner, inputs, outputs);
    this.uuid = uuid;
    this.mnemonic = mnemonic;
    this.infoExtension = infoExtension;
    this.info = info;
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException {
    throw new ActionExecutionException(
        mnemonic + "ExtraAction should not be executed.", this, false);
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  protected String computeKey() {
    return new Fingerprint()
        .addUUID(uuid)
        .addBytes(info.toByteArray())
        .hexDigestAndReset();
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return ResourceSet.ZERO;
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo() {
    return super.getExtraActionInfo().setExtension(infoExtension, info);
  }

  public static Artifact getDummyOutput(RuleContext ruleContext) {
    return ruleContext.getPackageRelativeArtifact(
        ruleContext.getLabel().getName() + ".extra_action_dummy",
        ruleContext.getConfiguration().getGenfilesDirectory());
  }
}
