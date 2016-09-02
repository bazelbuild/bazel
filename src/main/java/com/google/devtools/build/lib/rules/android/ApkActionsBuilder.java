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
  String apkName;
  boolean zipalignApk = false;

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

  /** Requests a signed APK be built at the specified artifact. */
  public ApkActionsBuilder setSignedApk(Artifact signedApk) {
    this.signedApk = signedApk;
    return this;
  }

  /** Requests that signed APKs are zipaligned. */
  public ApkActionsBuilder setZipalignApk(boolean zipalign) {
    this.zipalignApk = zipalign;
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

  /** Uses the zipalign tool to align the zip boundaries for uncompressed resources by 4 bytes. */
  Action[] zipalignApk(RuleContext ruleContext, Artifact inputApk, Artifact zipAlignedApk) {
    return new SpawnAction.Builder()
        .addInput(inputApk)
        .addOutput(zipAlignedApk)
        .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getZipalign())
        .addArgument("4")
        .addInputArgument(inputApk)
        .addOutputArgument(zipAlignedApk)
        .setProgressMessage("Zipaligning " + apkName)
        .setMnemonic("AndroidZipAlign")
        .build(ruleContext);
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

      if (signedApk != null) {
        // Legacy signing destroys zip aligning, so if zip aligning is requested we build an
        // intermediate APK that is signed but not zip aligned then zip align it. If zip aligning
        // is not requested then the output of the buildApk step is the final apk.
        Artifact intermediateSignedApk;
        if (zipalignApk) {
          intermediateSignedApk =
              AndroidBinary.getDxArtifact(ruleContext, "signed_" + signedApk.getFilename());
        } else {
          intermediateSignedApk = signedApk;
        }

        ruleContext.registerAction(buildApk(
            ruleContext,
            intermediateSignedApk,
            semantics.getApkDebugSigningKey(ruleContext),
            "Generating signed " + apkName));

        if (zipalignApk) {
          ruleContext.registerAction(
              zipalignApk(ruleContext, intermediateSignedApk, signedApk));
        }
      }
    }
  }

  /**
   * This implementation uses the executables from the apkbuilder and apksigner attributes of
   * android_sdk to build and sign the SDKs. Zipaligning is done before the apksigner
   * by the zipalign tool. The signer supports both V1 and V2 signing signatures configured by the
   * {@link ApkSigningMethod} parameter.
   */
  static class SignerToolApkActionsBuilder extends ApkActionsBuilder {
    private final ApkSigningMethod signingMethod;

    SignerToolApkActionsBuilder(ApkSigningMethod signingMethod) {
      this.signingMethod = signingMethod;
    }

    @Override
    public void registerActions(RuleContext ruleContext, AndroidSemantics semantics) {
      Preconditions.checkNotNull(
          apkName, "APK name must be set to create progress messages for APK actions.");

      if (signedApk != null) {
        Artifact intermediateUnsignedApk = unsignedApk;
        if (intermediateUnsignedApk == null) {
          // If the caller did not request an unsigned APK, we still need to construct one so that
          // we can sign it. So we make up an intermediate artifact.
          intermediateUnsignedApk =
              AndroidBinary.getDxArtifact(ruleContext, "unsigned_" + signedApk.getFilename());
        }
        ruleContext.registerAction(
            buildApk(ruleContext, intermediateUnsignedApk, null, "Generating unsigned " + apkName));

        Artifact apkToSign = intermediateUnsignedApk;
        if (zipalignApk) {
          apkToSign =
              AndroidBinary.getDxArtifact(ruleContext, "zipaligned_" + signedApk.getFilename());
          ruleContext.registerAction(zipalignApk(ruleContext, intermediateUnsignedApk, apkToSign));
        }

        ruleContext.registerAction(signApk(
            ruleContext,
            semantics.getApkDebugSigningKey(ruleContext),
            apkToSign,
            signedApk));
      } else if (unsignedApk != null) {
        ruleContext.registerAction(
            buildApk(ruleContext, unsignedApk, null, "Generating unsigned " + apkName));
      }
    }

    /**
     * Signs an APK using the ApkSignerTool. Supports both the jar signing scheme(v1) and the apk
     * signing scheme v2. Note that zip alignment is preserved by this step. Furthermore,
     * zip alignment cannot be performed after v2 signing without invalidating the signature.
     */
    private Action[] signApk(RuleContext ruleContext, Artifact signingKey, Artifact unsignedApk,
        Artifact signedAndZipalignedApk) {
      return new SpawnAction.Builder()
          .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getApkSigner())
          .setProgressMessage("Signing " + apkName)
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
