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
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.rules.android.apkmanifest.ApkManifestOuterClass;
import com.google.devtools.build.lib.rules.android.apkmanifest.ApkManifestOuterClass.ApkManifest;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.protobuf.ByteString;
import com.google.protobuf.TextFormat;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Map;

public class ApkManifestAction extends AbstractFileWriteAction {

  private static Iterable<Artifact> makeInputs(
      AndroidSdkProvider sdk,
      Iterable<Artifact> jars,
      Artifact resourceApk,
      NativeLibs nativeLibs) {

    return ImmutableList.<Artifact>builder()
        .add(sdk.getAapt().getExecutable())
        .add(sdk.getAdb().getExecutable())
        .add(sdk.getAidl().getExecutable())
        .add(sdk.getAndroidJar())
        .add(sdk.getAnnotationsJar())
        .add(sdk.getApkBuilder().getExecutable())
        .add(sdk.getDx().getExecutable())
        .add(sdk.getFrameworkAidl())
        .add(sdk.getJack().getExecutable())
        .add(sdk.getJill().getExecutable())
        .add(sdk.getMainDexClasses())
        .add(sdk.getMainDexListCreator().getExecutable())
        .add(sdk.getProguard().getExecutable())
        .add(sdk.getResourceExtractor().getExecutable())
        .add(sdk.getShrinkedAndroidJar())
        .add(sdk.getZipalign().getExecutable())
        
        .addAll(jars)
        .add(resourceApk)
        .addAll(nativeLibs.getAllNativeLibs())
        .build();
  }

  private static final String GUID = "7b6f4858-d1f2-11e5-83b0-cf6ddc5a32d9";

  private final boolean textOutput;
  private final AndroidSdkProvider sdk;
  private final Iterable<Artifact> jars;
  private final Artifact resourceApk;
  private final NativeLibs nativeLibs;

  public ApkManifestAction(
      ActionOwner owner,
      Artifact outputFile,
      boolean textOutput,
      AndroidSdkProvider sdk,
      Iterable<Artifact> jars,
      Artifact resourceApk,
      NativeLibs nativeLibs) {

    super(owner, makeInputs(sdk, jars, resourceApk, nativeLibs), outputFile, false);
    this.textOutput = textOutput;
    this.sdk = sdk;
    this.jars = jars;
    this.resourceApk = resourceApk;
    this.nativeLibs = nativeLibs;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) throws IOException {

    ApkManifestCreator manifestCreator = new ApkManifestCreator(
        ctx.getMetadataHandler());

    final ApkManifest manifest = manifestCreator.createManifest();

    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        if (textOutput) {
          TextFormat.print(manifest, new PrintStream(out));
        } else {
          manifest.writeTo(out);
        }
      }
    };
  }

  @Override
  protected String computeKey() {
    return new Fingerprint()
        .addString(GUID)
        .hexDigestAndReset();
  }

  private class ApkManifestCreator {

    private final MetadataHandler metadataHandler;

    private ApkManifestCreator(MetadataHandler metadataHandler) {
      this.metadataHandler = metadataHandler;
    }
 
    private ApkManifest createManifest() throws IOException {
      ApkManifest.Builder manifestBuilder = ApkManifest.newBuilder();

      for (Artifact jar : jars) {
        manifestBuilder.addJars(makeArtifactProto(jar));
      }

      manifestBuilder.setResourceApk(makeArtifactProto(resourceApk));

      for (Map.Entry<String, Iterable<Artifact>> nativeLib : nativeLibs.getMap().entrySet()) {
        if (!Iterables.isEmpty(nativeLib.getValue())) {
          manifestBuilder.addNativeLibBuilder()
              .setArch(nativeLib.getKey())
              .addAllNativeLibs(makeArtifactProtos(nativeLib.getValue()));
        }
      }

      manifestBuilder.setAndroidSdk(createAndroidSdk(sdk));
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
      byte[] digest = metadataHandler.getMetadata(artifact).digest;
      return ApkManifestOuterClass.Artifact.newBuilder()
          .setExecRootPath(artifact.getExecPathString())
          .setHash(ByteString.copyFrom(digest))
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
      sdkProto.setApkbuilder(getArtifactPath(sdk.getApkBuilder()));
      sdkProto.setDx(getArtifactPath(sdk.getDx()));
      sdkProto.setFrameworkAidl(getArtifactPath(sdk.getFrameworkAidl()));
      sdkProto.setJack(getArtifactPath(sdk.getJack()));
      sdkProto.setJill(getArtifactPath(sdk.getJill()));
      sdkProto.setMainDexClasses(getArtifactPath(sdk.getMainDexClasses()));
      sdkProto.setMainDexListCreator(getArtifactPath(sdk.getMainDexListCreator()));
      sdkProto.setProguard(getArtifactPath(sdk.getProguard()));
      sdkProto.setResourceExtractor(getArtifactPath(sdk.getResourceExtractor()));
      sdkProto.setShrinkedAndroidJar(getArtifactPath(sdk.getShrinkedAndroidJar()));
      sdkProto.setZipalign(getArtifactPath(sdk.getZipalign()));
      sdkProto.setBuildToolsVersion(sdk.getBuildToolsVersion());

      return sdkProto.build();
    }
  }
}
