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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.ApkSigningMethod;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A class for coordinating APK building, signing and zipaligning.
 *
 * <p>It is not always necessary to zip align APKs, for instance if the APK does not contain
 * resources. Furthermore, we do not always care about the unsigned apk because it cannot be
 * installed on a device until it is signed.
 */
public class ApkActionsBuilder {
  private Artifact classesDex;
  private ImmutableList.Builder<Artifact> inputZips = new ImmutableList.Builder<>();
  private Artifact javaResourceZip;
  private FilesToRunProvider resourceExtractor;
  private Artifact javaResourceFile;
  private NativeLibs nativeLibs = NativeLibs.EMPTY;
  private Artifact unsignedApk;
  private Artifact signedApk;
  private boolean zipalignApk = false;
  private Artifact signingKey;

  private final String apkName;

  public static ApkActionsBuilder create(String apkName) {
    return new ApkActionsBuilder(apkName);
  }

  private ApkActionsBuilder(String apkName) {
    this.apkName = apkName;
  }

  /** Sets the native libraries to be included in the APK. */
  public ApkActionsBuilder setNativeLibs(NativeLibs nativeLibs) {
    this.nativeLibs = nativeLibs;
    return this;
  }

  /**
   * Sets the dex file to be included in the APK.
   *
   * <p>Can be either a plain classes.dex or a .zip file containing dexes.
   */
  public ApkActionsBuilder setClassesDex(Artifact classesDex) {
    Preconditions.checkArgument(
        classesDex.getFilename().endsWith(".zip")
            || classesDex.getFilename().equals("classes.dex"));
    this.classesDex = classesDex;
    return this;
  }

  /** Add a zip file that should be copied as is into the APK. */
  public ApkActionsBuilder addInputZip(Artifact inputZip) {
    this.inputZips.add(inputZip);
    return this;
  }

  public ApkActionsBuilder addInputZips(Iterable<Artifact> inputZips) {
    this.inputZips.addAll(inputZips);
    return this;
  }

  /**
   * Adds a zip to be added to the APK and an executable that filters the zip to extract the
   * relevant contents first.
   */
  public ApkActionsBuilder setJavaResourceZip(
      Artifact javaResourceZip, FilesToRunProvider resourceExtractor) {
    this.javaResourceZip = javaResourceZip;
    this.resourceExtractor = resourceExtractor;
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

  /** Sets the signing key that will be used to sign the APK. */
  public ApkActionsBuilder setSigningKey(Artifact signingKey) {
    this.signingKey = signingKey;
    return this;
  }

  /** Registers the actions needed to build the requested APKs in the rule context. */
  public void registerActions(RuleContext ruleContext) {
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
      legacyBuildApk(ruleContext, intermediateUnsignedApk, "Generating unsigned " + apkName);
    }

    if (signedApk != null) {
      Artifact apkToSign = intermediateUnsignedApk;
      // Zipalignment is performed before signing. So if a zipaligned APK is requested, we need an
      // intermediate zipaligned-but-not-signed apk artifact.
      if (zipalignApk) {
        apkToSign =
            AndroidBinary.getDxArtifact(ruleContext, "zipaligned_" + signedApk.getFilename());
        zipalignApk(ruleContext, intermediateUnsignedApk, apkToSign);
      }
      signApk(ruleContext, apkToSign, signedApk);
    }
  }

  /**
   * Registers generating actions for {@code outApk} that builds the APK specified.
   *
   * <p>If {@code signingKey} is not null, the apk will be signed with it using the V1 signature
   * scheme.
   */
  private void legacyBuildApk(RuleContext ruleContext, Artifact outApk, String message) {
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

    Pair<Artifact, Runfiles> nativeSymlinksManifestAndRunfiles =
        nativeLibs.createApkBuilderSymlinks(ruleContext);
    if (nativeSymlinksManifestAndRunfiles != null) {
      Artifact nativeSymlinksManifest = nativeSymlinksManifestAndRunfiles.first;
      Runfiles nativeSymlinksRunfiles = nativeSymlinksManifestAndRunfiles.second;
      PathFragment nativeSymlinksDir = nativeSymlinksManifest.getExecPath().getParentDirectory();
      actionBuilder
          .addRunfilesSupplier(
              new RunfilesSupplierImpl(
                  nativeSymlinksDir,
                  nativeSymlinksRunfiles,
                  nativeSymlinksManifest))
          .addInput(nativeSymlinksManifest)
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

    actionBuilder.addArgument("-u");

    for (Artifact inputZip : inputZips.build()) {
      actionBuilder.addArgument("-z").addInputArgument(inputZip);
    }

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
    Artifact compressedApk =
        AndroidBinary.getDxArtifact(ruleContext, "compressed_" + outApk.getFilename());
    SpawnAction.Builder compressedApkActionBuilder = new SpawnAction.Builder()
        .setMnemonic("ApkBuilder")
        .setProgressMessage(message)
        .addArgument("--exclude_build_data")
        .addArgument("--compression")
        .addArgument("--normalize")
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
            .addInput(classesDex)
            .addArgument("--resources")
            .addArgument(
                singleJarResourcesArgument(
                    classesDex.getExecPathString(),
                    classesDex.getFilename()));
      }
    }

    if (javaResourceFile != null) {
      compressedApkActionBuilder
          .addInput(javaResourceFile)
          .addArgument("--resources")
          .addArgument(
              singleJarResourcesArgument(
                  javaResourceFile.getExecPathString(),
                  javaResourceFile.getFilename()));
    }

    for (String architecture : nativeLibs.getMap().keySet()) {
      for (Artifact nativeLib : nativeLibs.getMap().get(architecture)) {
        compressedApkActionBuilder
            .addArgument("--resources")
            .addArgument(
                singleJarResourcesArgument(
                    nativeLib.getExecPathString(),
                    "lib/" + architecture + "/" + nativeLib.getFilename()))
            .addInput(nativeLib);
      }
    }

    SpawnAction.Builder singleJarActionBuilder = new SpawnAction.Builder()
        .setMnemonic("ApkBuilder")
        .setProgressMessage(message)
        .addArgument("--exclude_build_data")
        .addArgument("--dont_change_compression")
        .addArgument("--normalize")
        .addArgument("--sources")
        .addInputArgument(compressedApk)
        .addArgument("--output")
        .addOutputArgument(outApk);
    setSingleJarAsExecutable(ruleContext, singleJarActionBuilder);

    if (javaResourceZip != null) {
      // The javaResourceZip contains many files that are unwanted in the APK such as .class files.
      Artifact extractedJavaResourceZip =
          AndroidBinary.getDxArtifact(ruleContext, "extracted_" + javaResourceZip.getFilename());
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .setExecutable(resourceExtractor)
              .setMnemonic("ResourceExtractor")
              .setProgressMessage("Extracting Java resources from deploy jar for %s", apkName)
              .addInputArgument(javaResourceZip)
              .addOutputArgument(extractedJavaResourceZip)
              .build(ruleContext));

      if (ruleContext.getFragment(AndroidConfiguration.class).compressJavaResources()) {
        compressedApkActionBuilder
            .addArgument("--sources")
            .addInputArgument(extractedJavaResourceZip);
      } else {
        singleJarActionBuilder
            .addArgument("--sources")
            .addInputArgument(extractedJavaResourceZip);
      }
    }

    if (nativeLibs.getName() != null) {
      singleJarActionBuilder
          .addArgument("--resources")
          .addArgument(
              singleJarResourcesArgument(
                  nativeLibs.getName().getExecPathString(),
                  nativeLibs.getName().getFilename()))
          .addInput(nativeLibs.getName());
    }

    for (Artifact inputZip : inputZips.build()) {
      singleJarActionBuilder.addArgument("--sources").addInputArgument(inputZip);
    }

    ImmutableList<String> noCompressExtensions =
        ruleContext.getTokenizedStringListAttr("nocompress_extensions");
    if (ruleContext.getFragment(AndroidConfiguration.class).useNocompressExtensionsOnApk()
        && !noCompressExtensions.isEmpty()) {
      compressedApkActionBuilder
          .addArgument("--nocompress_suffixes")
          .addArguments(noCompressExtensions);
      singleJarActionBuilder
          .addArgument("--nocompress_suffixes")
          .addArguments(noCompressExtensions);
    }

    ruleContext.registerAction(compressedApkActionBuilder.build(ruleContext));
    ruleContext.registerAction(singleJarActionBuilder.build(ruleContext));
  }

  /**
   * The --resources flag to singlejar can have either of the following forms:
   * <ul>
   * <li>The path to the input file. In this case the file is placed at the same path in the APK.
   * <li>{@code from}:{@code to} where {@code from} is that path to the input file and {@code to} is
   * the location in the APK to put it.
   * </ul>
   * This method creates the syntax for the second form.
   */
  private static String singleJarResourcesArgument(String from, String to) {
    return from + ":" + to;
  }

  /** Uses the zipalign tool to align the zip boundaries for uncompressed resources by 4 bytes. */
  private void zipalignApk(RuleContext ruleContext, Artifact inputApk, Artifact zipAlignedApk) {
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(inputApk)
            .addOutput(zipAlignedApk)
            .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getZipalign())
            .addArgument("4")
            .addInputArgument(inputApk)
            .addOutputArgument(zipAlignedApk)
            .setProgressMessage("Zipaligning %s", apkName)
            .setMnemonic("AndroidZipAlign")
            .build(ruleContext));
  }

  /**
   * Signs an APK using the ApkSignerTool. Supports both the jar signing scheme(v1) and the apk
   * signing scheme v2. Note that zip alignment is preserved by this step. Furthermore, zip
   * alignment cannot be performed after v2 signing without invalidating the signature.
   */
  private void signApk(
      RuleContext ruleContext, Artifact unsignedApk, Artifact signedAndZipalignedApk) {
    ApkSigningMethod signingMethod =
        ruleContext.getFragment(AndroidConfiguration.class).getApkSigningMethod();
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getApkSigner())
            .setProgressMessage("Signing %s", apkName)
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
              JavaCommon.getHostJavaExecutable(ruleContext),
              singleJar,
              JavaToolchainProvider.fromRuleContext(ruleContext).getJvmOptions())
          .addTransitiveInputs(JavaHelper.getHostJavabaseInputs(ruleContext));
    } else {
      builder.setExecutable(singleJar);
    }
  }
}
