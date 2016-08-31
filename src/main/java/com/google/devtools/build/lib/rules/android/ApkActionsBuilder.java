// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.ApkSigningMethod;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/**
 * A class for coordinating APK building, signing and zipaligning.
 *
 * <p>It is not always necessary to zip align APKs, for instance if the APK does not contain
 * resources. Furthermore, we do not always care about the unsigned apk because it cannot be
 * installed on a device until it is signed.
 *
 * <p>This interface allows the caller to specify their desired APKs and the implementations
 * determines which tools to use to build actions to fulfill them.
 */
public abstract class ApkActionsBuilder {
  private Artifact classesDex;
  private Artifact resourceApk;
  private Artifact javaResourceZip;
  private Artifact javaResourceFile;
  private NativeLibs nativeLibs = NativeLibs.EMPTY;

  Artifact unsignedApk;
  Artifact signedApk;
  Artifact signedAndZipalignedApk;
  String apkName;

  /** Sets the user-visible apkName that is included in the action progress messages. */
  public ApkActionsBuilder setApkName(String apkName) {
    this.apkName = apkName;
    return this;
  }

  /** Registers the actions needed to build the requested APKs in the rule context. */
  public abstract void registerActions(RuleContext ruleContext, AndroidSemantics semantics);

  /** Sets the native libraries to be included in the APK. */
  public ApkActionsBuilder setNativeLibs(NativeLibs nativeLibs) {
    this.nativeLibs = nativeLibs;
    return this;
  }

  /**
   * Sets the dex file to be included in the APK.
   *
   * <p>Can be either a plain .dex or a .zip file containing dexes.
   */
  public ApkActionsBuilder setClassesDex(Artifact classesDex) {
    this.classesDex = classesDex;
    return this;
  }

  /** Sets the resource APK that contains the Android resources to be bundled into the output. */
  public ApkActionsBuilder setResourceApk(Artifact resourceApk) {
    this.resourceApk = resourceApk;
    return this;
  }

  /**
   * Sets the file where Java resources are taken.
   *
   * <p>The contents of this zip will will be put directly into the APK except for files that are
   * filtered out by the {@link com.android.sdklib.build.ApkBuilder} which seem to not be resources,
   * e.g. files with the extension {@code .class}.
   */
  public ApkActionsBuilder setJavaResourceZip(Artifact javaResourceZip) {
    this.javaResourceZip = javaResourceZip;
    return this;
  }

  /**
   * Adds an individual resource file to the root directory of the APK.
   *
   * <p>This provides the same functionality as {@code javaResourceZip}, except much more hacky.
   * Will most probably won't work if there is an input artifact in the same directory as this
   * file.
   */
  public ApkActionsBuilder setJavaResourceFile(Artifact javaResourceFile) {
    this.javaResourceFile = javaResourceFile;
    return this;
  }

  /** Requests an unsigned APK be built at the specified artifact. */
  public ApkActionsBuilder setUnsignedApk(Artifact unsignedApk) {
    this.unsignedApk = unsignedApk;
    return this;
  }

  /** Requests a signed but not necessarily zip aligned APK be built at the specified artifact. */
  public ApkActionsBuilder setSignedApk(Artifact signedApk) {
    this.signedApk = signedApk;
    return this;
  }

  /** Requests a signed and zipaligned APK be built at the specified artifact. */
  public ApkActionsBuilder setSignedAndZipalignedApk(Artifact signedAndZipalignedApk) {
    this.signedAndZipalignedApk = signedAndZipalignedApk;
    return this;
  }

  /**
   * Creates a generating action for {@code outApk} that builds the APK specified.
   *
   * <p>If {@code signingKey} is not null, the apk will be signed with it using the V1 signature
   * scheme.
   */
  Action[] buildApk(RuleContext ruleContext, Artifact outApk, Artifact signingKey, String message) {
    SpawnAction.Builder actionBuilder = new SpawnAction.Builder()
        .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getApkBuilder())
        .setProgressMessage(message)
        .setMnemonic("AndroidApkBuilder")
        .addOutputArgument(outApk);

    if (javaResourceZip != null) {
      actionBuilder
          .addArgument("-rj")
          .addInputArgument(javaResourceZip);
    }

    Artifact nativeSymlinks = nativeLibs.createApkBuilderSymlinks(ruleContext);
    if (nativeSymlinks != null) {
      PathFragment nativeSymlinksDir = nativeSymlinks.getExecPath().getParentDirectory();
      actionBuilder
          .addInputManifest(nativeSymlinks, nativeSymlinksDir)
          .addInput(nativeSymlinks)
          .addInputs(nativeLibs.getAllNativeLibs())
          .addArgument("-nf")
          // If the native libs are "foo/bar/x86/foo.so", we need to pass "foo/bar" here
          .addArgument(nativeSymlinksDir.getPathString());
    }

    if (nativeLibs.getName() != null) {
      actionBuilder
          .addArgument("-rf")
          .addArgument(nativeLibs.getName().getExecPath().getParentDirectory().getPathString())
          .addInput(nativeLibs.getName());
    }

    if (javaResourceFile != null) {
      actionBuilder
          .addArgument("-rf")
          .addArgument((javaResourceFile.getExecPath().getParentDirectory().getPathString()))
          .addInput(javaResourceFile);
    }

    if (signingKey == null) {
      actionBuilder.addArgument("-u");
    } else {
      actionBuilder.addArgument("-ks").addArgument(signingKey.getExecPathString());
      actionBuilder.addInput(signingKey);
    }

    actionBuilder
        .addArgument("-z")
        .addInputArgument(resourceApk);

    if (classesDex != null) {
      actionBuilder
          .addArgument(classesDex.getFilename().endsWith(".dex") ? "-f" : "-z")
          .addInputArgument(classesDex);
    }

    return actionBuilder.build(ruleContext);
  }

  /**
   * An implementation that uses ApkBuilderMain to both build and sign APKs and the Android SDK
   * zipalign tool to zipalign. This implementation only supports V1 signature scheme (JAR signing).
   */
  static class LegacySignerApkActionsBuilder extends ApkActionsBuilder {

    @Override
    public void registerActions(RuleContext ruleContext, AndroidSemantics semantics) {
      Preconditions.checkNotNull(
          apkName, "APK name must be set to create progress messages for APK actions.");

      if (unsignedApk != null) {
        ruleContext.registerAction(
            buildApk(ruleContext, unsignedApk, null, "Generating unsigned " + apkName));
      }

      if (signedAndZipalignedApk != null) {
        Artifact intermediateSignedApk = this.signedApk;
        if (intermediateSignedApk == null) {
          // If the caller requested a zipaligned APK but not a signed APK, we still need to build
          // a signed APK as an intermediate so we construct an artifact.
          intermediateSignedApk = AndroidBinary.getDxArtifact(
              ruleContext, "signed_" + signedAndZipalignedApk.getFilename());
        }
        ruleContext.registerAction(buildApk(
            ruleContext,
            intermediateSignedApk,
            semantics.getApkDebugSigningKey(ruleContext),
            "Generating signed " + apkName));
        ruleContext.registerAction(
            zipalignApk(ruleContext, intermediateSignedApk, signedAndZipalignedApk));
      } else if (signedApk != null) {
        ruleContext.registerAction(buildApk(
            ruleContext,
            signedApk,
            semantics.getApkDebugSigningKey(ruleContext),
            "Generating signed " + apkName));
      }
    }

    /** Last step in buildings an apk: align the zip boundaries by 4 bytes. */
    private Action[] zipalignApk(RuleContext ruleContext, Artifact signedApk,
        Artifact zipAlignedApk) {
      List<String> args = new ArrayList<>();
      // "4" is the only valid value for zipalign, according to:
      // http://developer.android.com/guide/developing/tools/zipalign.html
      args.add("4");
      args.add(signedApk.getExecPathString());
      args.add(zipAlignedApk.getExecPathString());

      return new SpawnAction.Builder()
          .addInput(signedApk)
          .addOutput(zipAlignedApk)
          .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getZipalign())
          .addArguments(args)
          .setProgressMessage("Zipaligning " + apkName)
          .setMnemonic("AndroidZipAlign")
          .build(ruleContext);
    }
  }

  /**
   * This implementation uses the executables from the apkbuilder and apksigner attributes of
   * android_sdk to build and sign the SDKs. Zipaligning is done by the apksigner
   * {@link com.android.apksigner.ApkSignerTool} between the V1 and V2 signature steps. The signer
   * supports both V1 and V2 signing signatures configured by the {@link ApkSigningMethod}
   * parameter.
   */
  static class SignerToolApkActionsBuilder extends ApkActionsBuilder {
    private final ApkSigningMethod signingMethod;

    SignerToolApkActionsBuilder(ApkSigningMethod signingMethod) {
      this.signingMethod = signingMethod;
    }

    @Override
    public void registerActions(RuleContext ruleContext, AndroidSemantics semantics) {
      // Only one should ever be specified. The only reason that both options exist are as a slight
      // optimization for the legacy code path in which we only zip align full APKs, not split APKs.
      Preconditions.checkState(
          signedApk == null || signedAndZipalignedApk == null,
          "ApkSignerTool cannot generate separate signedApk and signedAndZipalignedApk because "
              + "zipaligning is done between the v1 signing and v2 signing in the same action.");
      Preconditions.checkNotNull(
          apkName, "APK name must be set to create progress messages for APK actions.");

      Artifact finalApk = signedApk == null ? signedAndZipalignedApk : signedApk;
      if (finalApk != null) {
        Artifact intermediateUnsignedApk = this.unsignedApk;
        if (intermediateUnsignedApk == null) {
          // If the caller did not request an unsigned APK, we still need to construct one so that
          // we can sign it. So we make up an intermediate artifact.
          intermediateUnsignedApk = AndroidBinary.getDxArtifact(
              ruleContext, "unsigned_" + finalApk.getFilename());
        }
        ruleContext.registerAction(
            buildApk(ruleContext, intermediateUnsignedApk, null, "Generating unsigned " + apkName));
        ruleContext.registerAction(sign(
            ruleContext,
            semantics.getApkDebugSigningKey(ruleContext),
            intermediateUnsignedApk,
            finalApk));
      } else if (unsignedApk != null) {
        ruleContext.registerAction(
            buildApk(ruleContext, unsignedApk, null, "Generating unsigned " + apkName));
      }
    }

    /**
     * Signs and zip aligns an APK using the ApkSignerTool. Supports both the jar signing schema
     * (v1) and the apk signing schema v2.
     */
    private Action[] sign(RuleContext ruleContext, Artifact signingKey, Artifact unsignedApk,
        Artifact signedAndZipalignedApk) {
      return new SpawnAction.Builder()
          .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getApkSigner())
          .setProgressMessage("Signing and zipaligning " + apkName)
          .setMnemonic("ApkSignerTool")
          .addArgument("sign")
          .addArgument("--ks")
          .addInputArgument(signingKey)
          .addArguments("--ks-pass", "pass:android")
          .addArguments("--v1-signing-enabled", Boolean.toString(signingMethod.signV1()))
          .addArguments("--v1-signer-name", "CERT")
          .addArguments("--v2-signing-enabled", Boolean.toString(signingMethod.signV2()))
          .addArgument("--out")
          .addOutputArgument(signedAndZipalignedApk)
          .addInputArgument(unsignedApk)
          .build(ruleContext);
    }
  }
}
