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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.primitives.Ints;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.ProtoDeterministicWriter;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.android.apkmanifest.ApkManifestOuterClass;
import com.google.devtools.build.lib.rules.android.apkmanifest.ApkManifestOuterClass.ApkManifest;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.ByteString;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Map;

@Immutable
public final class ApkManifestAction extends AbstractFileWriteAction {

  private static Iterable<Artifact> makeInputs(
      AndroidSdkProvider sdk,
      Iterable<Artifact> jars,
      ResourceApk resourceApk,
      NativeLibs nativeLibs,
      Artifact debugKeystore) {

    return ImmutableList.<Artifact>builder()
        .add(sdk.getAapt().getExecutable())
        .add(sdk.getAdb().getExecutable())
        .add(sdk.getAidl().getExecutable())
        .add(sdk.getAndroidJar())
        .add(sdk.getAnnotationsJar())
        .add(sdk.getDx().getExecutable())
        .add(sdk.getFrameworkAidl())
        .add(sdk.getMainDexClasses())
        .add(sdk.getMainDexListCreator().getExecutable())
        .add(sdk.getProguard().getExecutable())
        .add(sdk.getShrinkedAndroidJar())
        .add(sdk.getZipalign().getExecutable())
        .addAll(jars)
        .add(resourceApk.getArtifact())
        .add(resourceApk.getManifest())
        .addAll(nativeLibs.getAllNativeLibs())
        .add(debugKeystore)
        .build();
  }

  private static final String GUID = "7b6f4858-d1f2-11e5-83b0-cf6ddc5a32d9";

  private final boolean textOutput;
  private final AndroidSdkProvider sdk;
  private final Iterable<Artifact> jars;
  private final ResourceApk resourceApk;
  private final NativeLibs nativeLibs;
  private final Artifact debugKeystore;

  /**
   * @param owner The action owner.
   * @param outputFile The artifact to write the proto to.
   * @param textOutput Whether to write the output as a text proto.
   * @param sdk The Android SDK.
   * @param jars All the jars that would be merged and dexed and put into an APK.
   * @param resourceApk The ResourceApk for the .ap_ that contains the resources that would go into
   *     an APK.
   * @param debugKeystore The debug keystore.
   * @param nativeLibs The natives libs that would go into an APK.
   */
  public ApkManifestAction(
      ActionOwner owner,
      Artifact outputFile,
      boolean textOutput,
      AndroidSdkProvider sdk,
      Iterable<Artifact> jars,
      ResourceApk resourceApk,
      NativeLibs nativeLibs,
      Artifact debugKeystore) {
    super(owner, makeInputs(sdk, jars, resourceApk, nativeLibs, debugKeystore), outputFile, false);
    CollectionUtils.checkImmutable(jars);
    this.textOutput = textOutput;
    this.sdk = sdk;
    this.jars = jars;
    this.resourceApk = resourceApk;
    this.nativeLibs = nativeLibs;
    this.debugKeystore = debugKeystore;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(final ActionExecutionContext ctx)
      throws IOException {

    ApkManifestCreator manifestCreator = new ApkManifestCreator(new ArtifactDigester() {
        @Override
        public byte[] getDigest(Artifact artifact) throws IOException {
          return ctx.getMetadataHandler().getMetadata(artifact).getDigest();
        }
    });

    final ApkManifest manifest = manifestCreator.createManifest();

    if (textOutput) {
      return new DeterministicWriter() {
        @Override
        public void writeOutputFile(OutputStream out) throws IOException {
          TextFormat.print(manifest, new PrintStream(out));
        }
      };
    } else {
      return new ProtoDeterministicWriter(manifest);
    }
  }

  @Override
  protected String computeKey() {

    // Use fake hashes for the purposes of the action's key, because the hashes are retrieved from
    // the MetadataHandler, which is available at only action-execution time. This should be ok
    // because if an input artifact changes (and hence its hash changes), the action should be rerun
    // anyway. This is more for the purpose of putting the structure of the output data into the
    // key.
    ApkManifestCreator manifestCreator = new ApkManifestCreator(new ArtifactDigester() {
        @Override
        public byte[] getDigest(Artifact artifact) {
          return Ints.toByteArray(artifact.getExecPathString().hashCode());
        }
    });

    ApkManifest manifest;
    try {
      manifest = manifestCreator.createManifest();
    } catch (IOException e) {
      // The ArtifactDigester shouldn't actually throw IOException, that's just for the
      // ArtifactDigester that uses the MetadataHandler.
      throw new IllegalStateException(e);
    }

    return new Fingerprint()
        .addString(GUID)
        .addBoolean(textOutput)
        .addBytes(manifest.toByteArray())
        .hexDigestAndReset();
  }

  private interface ArtifactDigester {
    byte[] getDigest(Artifact artifact) throws IOException;
  }

  private class ApkManifestCreator {

    private final ArtifactDigester artifactDigester;

    private ApkManifestCreator(ArtifactDigester artifactDigester) {
      this.artifactDigester = artifactDigester;
    }

    private ApkManifest createManifest() throws IOException {
      ApkManifest.Builder manifestBuilder = ApkManifest.newBuilder();

      for (Artifact jar : jars) {
        manifestBuilder.addJars(makeArtifactProto(jar));
      }

      manifestBuilder.setResourceApk(makeArtifactProto(resourceApk.getArtifact()));
      manifestBuilder.setAndroidManifest(makeArtifactProto(resourceApk.getManifest()));

      for (Map.Entry<String, NestedSet<Artifact>> nativeLib : nativeLibs.getMap().entrySet()) {
        if (!Iterables.isEmpty(nativeLib.getValue())) {
          manifestBuilder.addNativeLibBuilder()
              .setArch(nativeLib.getKey())
              .addAllNativeLibs(makeArtifactProtos(nativeLib.getValue()));
        }
      }

      manifestBuilder.setAndroidSdk(createAndroidSdk(sdk));
      manifestBuilder.setDebugKeystore(makeArtifactProto(debugKeystore));
      return manifestBuilder.build();
    }

    private Iterable<ApkManifestOuterClass.Artifact> makeArtifactProtos(
        Iterable<Artifact> artifacts) throws IOException {

      ImmutableList.Builder<ApkManifestOuterClass.Artifact> protoArtifacts =
          ImmutableList.builder();
      for (Artifact artifact : artifacts) {
        protoArtifacts.add(makeArtifactProto(artifact));
      }
      return protoArtifacts.build();
    }

    private ApkManifestOuterClass.Artifact makeArtifactProto(Artifact artifact) throws IOException {
      byte[] digest = artifactDigester.getDigest(artifact);
      return ApkManifestOuterClass.Artifact.newBuilder()
          .setExecRootPath(artifact.getExecPathString())
          .setHash(ByteString.copyFrom(digest))
          .setLabel(artifact.getOwnerLabel().toString())
          .build();
    }

    private String getArtifactPath(Artifact artifact) {
      return artifact.getExecPathString();
    }

    private String getArtifactPath(FilesToRunProvider filesToRunProvider) {
      return filesToRunProvider.getExecutable().getExecPathString();
    }

    private ApkManifestOuterClass.AndroidSdk createAndroidSdk(AndroidSdkProvider sdk) {

      ApkManifestOuterClass.AndroidSdk.Builder sdkProto =
          ApkManifestOuterClass.AndroidSdk.newBuilder();

      sdkProto.setAapt(getArtifactPath(sdk.getAapt()));
      sdkProto.setAdb(getArtifactPath(sdk.getAdb()));
      sdkProto.setAidl(getArtifactPath(sdk.getAidl()));
      sdkProto.setAndroidJar(getArtifactPath(sdk.getAndroidJar()));
      sdkProto.setAnnotationsJar(getArtifactPath(sdk.getAnnotationsJar()));
      sdkProto.setDx(getArtifactPath(sdk.getDx()));
      sdkProto.setFrameworkAidl(getArtifactPath(sdk.getFrameworkAidl()));
      sdkProto.setMainDexClasses(getArtifactPath(sdk.getMainDexClasses()));
      sdkProto.setMainDexListCreator(getArtifactPath(sdk.getMainDexListCreator()));
      sdkProto.setProguard(getArtifactPath(sdk.getProguard()));
      sdkProto.setShrinkedAndroidJar(getArtifactPath(sdk.getShrinkedAndroidJar()));
      sdkProto.setZipalign(getArtifactPath(sdk.getZipalign()));
      sdkProto.setBuildToolsVersion(sdk.getBuildToolsVersion());

      return sdkProto.build();
    }
  }
}
