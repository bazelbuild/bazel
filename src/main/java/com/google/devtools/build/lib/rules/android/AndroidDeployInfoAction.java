// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.android;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ByteStringDeterministicWriter;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.android.deployinfo.AndroidDeployInfoOuterClass;
import com.google.devtools.build.lib.rules.android.deployinfo.AndroidDeployInfoOuterClass.AndroidDeployInfo;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.IOException;
import java.io.InputStream;

/**
 * Writes AndroidDeployInfo proto message. This proto describes how to deploy and launch an
 * android_binary/android_test.
 */
@Immutable
public final class AndroidDeployInfoAction extends AbstractFileWriteAction {

  private static NestedSet<Artifact> makeInputs(
      Artifact mergedManifest,
      Iterable<Artifact> additionalMergedManifests,
      Iterable<Artifact> apksToDeploy) {
    return NestedSetBuilder.<Artifact>stableOrder()
        .add(mergedManifest)
        .addAll(additionalMergedManifests)
        .addAll(apksToDeploy)
        .build();
  }

  private static final String GUID = "eda283ba-9000-4b80-8dc4-7939101c44ba";

  private final Artifact mergedManifest;
  private final ImmutableList<Artifact> additionalMergedManifests;
  private final ImmutableList<Artifact> apksToDeploy;

  AndroidDeployInfoAction(
      ActionOwner owner,
      Artifact outputFile,
      Artifact mergedManifest,
      ImmutableList<Artifact> additionalMergedManifests,
      ImmutableList<Artifact> apksToDeploy) {
    super(
        owner,
        makeInputs(mergedManifest, additionalMergedManifests, apksToDeploy),
        outputFile,
        false);
    this.mergedManifest = mergedManifest;
    this.additionalMergedManifests = additionalMergedManifests;
    this.apksToDeploy = apksToDeploy;
  }

  private ByteString getByteString() {
    AndroidDeployInfoOuterClass.AndroidDeployInfo.Builder builder =
        AndroidDeployInfoOuterClass.AndroidDeployInfo.newBuilder();
    builder.setMergedManifest(makeArtifactProto(mergedManifest));
    for (Artifact additionMergedManifest : additionalMergedManifests) {
      builder.addAdditionalMergedManifests(makeArtifactProto(additionMergedManifest));
    }
    for (Artifact apk : apksToDeploy) {
      builder.addApksToDeploy(makeArtifactProto(apk));
    }
    return builder.build().toByteString();
  }

  static void createDeployInfoAction(
      RuleContext ruleContext,
      Artifact deployInfo,
      Artifact mergedManifest,
      ImmutableList<Artifact> additionalMergedManifests,
      ImmutableList<Artifact> apksToDeploy) {
    Action action =
        new AndroidDeployInfoAction(
            ruleContext.getActionOwner(),
            deployInfo,
            mergedManifest,
            additionalMergedManifests,
            apksToDeploy);
    ruleContext.registerAction(action);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) throws IOException {
    return new ByteStringDeterministicWriter(getByteString());
  }

  @VisibleForTesting
  public AndroidDeployInfo getDeployInfo() throws InvalidProtocolBufferException {
    return AndroidDeployInfo.parseFrom(getByteString());
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);

    try (InputStream in = getByteString().newInput()) {
      byte[] buffer = new byte[512];
      int amountRead;
      while ((amountRead = in.read(buffer)) != -1) {
        fp.addBytes(buffer, 0, amountRead);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static AndroidDeployInfoOuterClass.Artifact makeArtifactProto(Artifact artifact) {
    return AndroidDeployInfoOuterClass.Artifact.newBuilder()
        .setExecRootPath(artifact.getExecPathString())
        .build();
  }
}
