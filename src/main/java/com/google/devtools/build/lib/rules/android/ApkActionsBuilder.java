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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.ApkSigningMethod;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;

/**
 * A class for coordinating APK building, signing and zipaligning.
 *
 * <p>It is not always necessary to zip align APKs, for instance if the APK does not contain
 * resources. Furthermore, we do not always care about the unsigned apk because it cannot be
 * installed on a device until it is signed.
 */
public class ApkActionsBuilder {
  private Artifact classesDex;
  private Artifact resourceApk;
  private Artifact javaResourceZip;
  private Artifact javaResourceFile;
  private NativeLibs nativeLibs = NativeLibs.EMPTY;
  private Artifact unsignedApk;
  private Artifact signedApk;
  private boolean zipalignApk = false;

  private final String apkName;
  private final ApkSigningMethod signingMethod;

  public static ApkActionsBuilder create(String apkName, ApkSigningMethod signingMethod) {
    return new ApkActionsBuilder(apkName, signingMethod);
  }

  private ApkActionsBuilder(String apkName, ApkSigningMethod signingMethod) {
    this.apkName = apkName;
    this.signingMethod = signingMethod;
  }

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

  /** Registers the actions needed to build the requested APKs in the rule context. */
  public void registerActions(RuleContext ruleContext, AndroidSemantics semantics) {
    boolean useSingleJarApkBuilder =
        ruleContext.getFragment(AndroidConfiguration.class).useSingleJarApkBuilder();

    // If the caller did not request an unsigned APK, we still need to construct one so that
    // we can sign it. So we make up an intermediate artifact.
    Artifact intermediateUnsignedApk = unsignedApk != null
        ? unsignedApk
        : AndroidBinary.getDxArtifact(ruleContext, "unsigned_" + signedApk.getFilename());
    if (useSingleJarApkBuilder) {
      buildApk(ruleContext, intermediateUnsignedApk, "Generating unsigned " + apkName);
    } else {
      legacyBuildApk(ruleContext, intermediateUnsignedApk, null, "Generating unsigned " + apkName);
    }

    if (signedApk != null) {
      if (signingMethod.signLegacy()) {
        // With the legacy signer, zipalignment is performed after signing. So if a zipaligned APK
        // is requested, we need an intermediate signed-but-not-zipaligned apk artifact.
        Artifact intermediateSignedApk = zipalignApk
            ? AndroidBinary.getDxArtifact(ruleContext, "signed_" + signedApk.getFilename())
            : signedApk;
        legacyBuildApk(
            ruleContext,
            intermediateSignedApk,
            semantics.getApkDebugSigningKey(ruleContext),
            "Generating signed " + apkName);
        if (zipalignApk) {
          zipalignApk(ruleContext, intermediateSignedApk, signedApk);
        }
      } else {
        Artifact apkToSign = intermediateUnsignedApk;
        // With apksigner, zipalignment is performed before signing. So if a zipaligned APK is
        // requested, we need an intermediate zipaligned-but-not-signed apk artifact.
        if (zipalignApk) {
          apkToSign =
              AndroidBinary.getDxArtifact(ruleContext, "zipaligned_" + signedApk.getFilename());
          zipalignApk(ruleContext, intermediateUnsignedApk, apkToSign);
        }
        signApk(ruleContext, semantics.getApkDebugSigningKey(ruleContext), apkToSign, signedApk);
      }
    }
  }

  /**
   * Registers generating actions for {@code outApk} that builds the APK specified.
   *
   * <p>If {@code signingKey} is not null, the apk will be signed with it using the V1 signature
   * scheme.
   */
  private void legacyBuildApk(RuleContext ruleContext, Artifact outApk, Artifact signingKey,
      String message) {
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

    ruleContext.registerAction(actionBuilder.build(ruleContext));
  }

  /**
   * Registers generating actions for {@code outApk} that build an unsigned APK using SingleJar.
   */
  private void buildApk(RuleContext ruleContext, Artifact outApk, String message) {
    Map<String, String> executionInfo = ImmutableMap.of("supports-workers", "1");

    Artifact compressedApk =
        AndroidBinary.getDxArtifact(ruleContext, "compressed_" + outApk.getFilename());
    SpawnAction.Builder compressedApkActionBuilder = new SpawnAction.Builder()
        .setMnemonic("ApkBuilder")
        .setProgressMessage(message)
        .setExecutionInfo(executionInfo)
        .addArgument("--exclude_build_data")
        .addArgument("--compression")
        .addArgument("--output")
        .addOutputArgument(compressedApk);
    setSingleJarAsExecutable(ruleContext, compressedApkActionBuilder);

    if (classesDex != null) {
      if (classesDex.getFilename().endsWith(".zip")) {
        compressedApkActionBuilder
            .addArgument("--sources")
            .addInputArgument(classesDex);
      } else {
        compressedApkActionBuilder
            .addArgument("--resources")
            .addInputArgument(classesDex);
      }
    }

    if (javaResourceFile != null) {
      compressedApkActionBuilder
          .addArgument("--resources")
          .addInputArgument(javaResourceFile);
    }

    for (String architecture : nativeLibs.getMap().keySet()) {
      for (Artifact nativeLib : nativeLibs.getMap().get(architecture)) {
        compressedApkActionBuilder
            .addArgument("--resources")
            .addArgument(
                nativeLib.getExecPathString()
                    + ":lib/"
                    + architecture
                    + "/"
                    + nativeLib.getFilename())
            .addInput(nativeLib);
      }
    }

    ruleContext.registerAction(compressedApkActionBuilder.build(ruleContext));

    SpawnAction.Builder singleJarActionBuilder = new SpawnAction.Builder()
        .setMnemonic("ApkBuilder")
        .setProgressMessage(message)
        .setExecutionInfo(executionInfo)
        .addArgument("--exclude_build_data")
        .addArgument("--dont_change_compression")
        .addArgument("--sources")
        .addInputArgument(compressedApk)
        .addArgument("--output")
        .addOutputArgument(outApk);
    setSingleJarAsExecutable(ruleContext, singleJarActionBuilder);

    if (javaResourceZip != null) {
      // The javaResourceZip contains many files that are unwanted in the APK such as .class files.
      Artifact extractedJavaResourceZip =
          AndroidBinary.getDxArtifact(ruleContext, "extracted_" + javaResourceZip.getFilename());
      ruleContext.registerAction(new SpawnAction.Builder()
          .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getResourceExtractor())
          .setMnemonic("ResourceExtractor")
          .setProgressMessage("Extracting Java resources from deploy jar for " + apkName)
          .addInputArgument(javaResourceZip)
          .addOutputArgument(extractedJavaResourceZip)
          .build(ruleContext));

      singleJarActionBuilder
          .addArgument("--sources")
          .addInputArgument(extractedJavaResourceZip);
    }

    if (nativeLibs.getName() != null) {
      singleJarActionBuilder
          .addArgument("--resources")
          .addArgument(
              nativeLibs.getName().getExecPathString() + ":" + nativeLibs.getName().getFilename())
          .addInput(nativeLibs.getName());
    }

    if (resourceApk != null) {
      singleJarActionBuilder
          .addArgument("--sources")
          .addInputArgument(resourceApk);
    }
    ruleContext.registerAction(singleJarActionBuilder.build(ruleContext));
  }

  /** Uses the zipalign tool to align the zip boundaries for uncompressed resources by 4 bytes. */
  private void zipalignApk(RuleContext ruleContext, Artifact inputApk, Artifact zipAlignedApk) {
    ruleContext.registerAction(new SpawnAction.Builder()
        .addInput(inputApk)
        .addOutput(zipAlignedApk)
        .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getZipalign())
        .addArgument("4")
        .addInputArgument(inputApk)
        .addOutputArgument(zipAlignedApk)
        .setProgressMessage("Zipaligning " + apkName)
        .setMnemonic("AndroidZipAlign")
        .build(ruleContext));
  }

  /**
   * Signs an APK using the ApkSignerTool. Supports both the jar signing scheme(v1) and the apk
   * signing scheme v2. Note that zip alignment is preserved by this step. Furthermore,
   * zip alignment cannot be performed after v2 signing without invalidating the signature.
   */
  private void signApk(RuleContext ruleContext, Artifact signingKey,
      Artifact unsignedApk, Artifact signedAndZipalignedApk) {
    ruleContext.registerAction(new SpawnAction.Builder()
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
        .build(ruleContext));
  }

  // Adds the appropriate SpawnAction options depending on if SingleJar is a jar or not.
  private static void setSingleJarAsExecutable(
      RuleContext ruleContext, SpawnAction.Builder builder) {
    Artifact singleJar = JavaToolchainProvider.fromRuleContext(ruleContext).getSingleJar();
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .setJarExecutable(
              ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable(),
              singleJar,
              JavaToolchainProvider.fromRuleContext(ruleContext).getJvmOptions())
          .addTransitiveInputs(JavaHelper.getHostJavabaseInputs(ruleContext));
    } else {
      builder.setExecutable(singleJar);
    }
  }
}
