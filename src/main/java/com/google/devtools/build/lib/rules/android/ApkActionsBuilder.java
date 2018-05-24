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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction.Builder;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.ApkSigningMethod;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaRuntimeInfo;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

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
  private String artifactLocation;

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
   * Will most probably won't work if there is an input artifact in the same directory as this file.
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

  /** Sets the output APK instead of creating with a static/standard path. */
  public ApkActionsBuilder setArtifactLocationDirectory(String artifactLocation) {
    this.artifactLocation = artifactLocation;
    return this;
  }

  /** Registers the actions needed to build the requested APKs in the rule context. */
  public void registerActions(RuleContext ruleContext) {
    boolean useSingleJarApkBuilder =
        ruleContext.getFragment(AndroidConfiguration.class).useSingleJarApkBuilder();

    // If the caller did not request an unsigned APK, we still need to construct one so that
    // we can sign it. So we make up an intermediate artifact.
    Artifact intermediateUnsignedApk =
        unsignedApk != null
            ? unsignedApk
            : getApkArtifact(ruleContext, "unsigned_" + signedApk.getFilename());
    if (useSingleJarApkBuilder) {
      buildApk(ruleContext, intermediateUnsignedApk);
    } else {
      legacyBuildApk(ruleContext, intermediateUnsignedApk);
    }

    if (signedApk != null) {
      Artifact apkToSign = intermediateUnsignedApk;
      // Zipalignment is performed before signing. So if a zipaligned APK is requested, we need an
      // intermediate zipaligned-but-not-signed apk artifact.
      if (zipalignApk) {
        apkToSign = getApkArtifact(ruleContext, "zipaligned_" + signedApk.getFilename());
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
  private void legacyBuildApk(RuleContext ruleContext, Artifact outApk) {
    SpawnAction.Builder actionBuilder =
        new SpawnAction.Builder()
            .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getApkBuilder())
            .setProgressMessage("Generating unsigned %s with legacy apkbuilder", apkName)
            .setMnemonic("AndroidApkBuilder")
            .addOutput(outApk);
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder().addExecPath(outApk);

    if (javaResourceZip != null) {
      actionBuilder.addInput(javaResourceZip);
      commandLine.add("-rj").addExecPath(javaResourceZip);
    }

    NativeLibs.ManifestAndRunfiles nativeSymlinksManifestAndRunfiles =
        nativeLibs.createApkBuilderSymlinks(ruleContext);
    if (nativeSymlinksManifestAndRunfiles != null) {
      // This following is equal to AndroidBinary.getDxArtifact(
      //     ruleContext, "native_symlinks/MANIFEST").getExecPath().getParentDirectory();
      // However, that causes an artifact to be registered without a generating action under
      // --nobuild_runfile_manifests, so instead, the following directly synthesizes the required
      // path fragment.
      PathFragment nativeSymlinksDir =
          ruleContext
              .getBinOrGenfilesDirectory()
              .getExecPath()
              .getRelative(ruleContext.getUniqueDirectory("_dx").getRelative("native_symlinks"));

      actionBuilder
          .addRunfilesSupplier(
              new RunfilesSupplierImpl(
                  nativeSymlinksDir,
                  nativeSymlinksManifestAndRunfiles.runfiles,
                  nativeSymlinksManifestAndRunfiles.manifest))
          .addInputs(nativeLibs.getAllNativeLibs());
      if (nativeSymlinksManifestAndRunfiles.manifest != null) {
        actionBuilder.addInput(nativeSymlinksManifestAndRunfiles.manifest);
      }
      commandLine
          .add("-nf")
          // If the native libs are "foo/bar/x86/foo.so", we need to pass "foo/bar" here
          .addPath(nativeSymlinksDir);
    }

    if (nativeLibs.getName() != null) {
      actionBuilder.addInput(nativeLibs.getName());
      commandLine.add("-rf").addPath(nativeLibs.getName().getExecPath().getParentDirectory());
    }

    if (javaResourceFile != null) {
      actionBuilder.addInput(javaResourceFile);
      commandLine.add("-rf").addPath(javaResourceFile.getExecPath().getParentDirectory());
    }

    commandLine.add("-u");

    for (Artifact inputZip : inputZips.build()) {
      actionBuilder.addInput(inputZip);
      commandLine.addExecPath("-z", inputZip);
    }

    if (classesDex != null) {
      actionBuilder.addInput(classesDex);
      if (classesDex.getFilename().endsWith(".dex")) {
        commandLine.add("-f");
      } else {
        commandLine.add("-z");
      }
      commandLine.addExecPath(classesDex);
    }

    actionBuilder.addCommandLine(commandLine.build());
    ruleContext.registerAction(actionBuilder.build(ruleContext));
  }

  /** Registers generating actions for {@code outApk} that build an unsigned APK using SingleJar.
   *
   * <p>Depending on the flag --android_compress_apk_with_singlejar (default to true), this method
   * either:
   *
   * <ol>
   * <li>Builds the compressed APK, then builds the unsigned APK from that (2 actions, smaller APK)
   * <li>Builds the unsigned APK directly (1 action,  bigger APK)
   * </ol>
   *
   * <p>Note that these actions are *in the critical path* of any android_binary build, so that
   * extra APK generation action will incur a non-trivial overhead.
   */
  private void buildApk(RuleContext ruleContext, Artifact outApk) {

    SpawnAction.Builder singleJarActionBuilder =
        new SpawnAction.Builder()
            .setMnemonic("ApkBuilder")
            .setProgressMessage("Generating unsigned %s", apkName)
            .addOutput(outApk);
    setSingleJarAsExecutable(ruleContext, singleJarActionBuilder);

    CustomCommandLine.Builder singleJarCommandLine = CustomCommandLine.builder();
    singleJarCommandLine
        .add("--exclude_build_data")
        .add("--dont_change_compression")
        .add("--normalize")
        .addExecPath("--output", outApk);

    Artifact extractedJavaResourceZip = null;

    if (ruleContext.getFragment(AndroidConfiguration.class).getCompressApkWithSinglejar()) {

      Artifact compressedApk = getApkArtifact(ruleContext, "compressed_" + outApk.getFilename());
      SpawnAction.Builder compressedApkActionBuilder =
          new SpawnAction.Builder()
              .setMnemonic("ApkBuilder")
              .setProgressMessage("Generating compressed unsigned %s", apkName)
              .addOutput(compressedApk);
      CustomCommandLine.Builder compressedApkCommandLine =
          CustomCommandLine.builder()
              .add("--exclude_build_data")
              .add("--compression")
              .add("--normalize")
              .addExecPath("--output", compressedApk);
      setSingleJarAsExecutable(ruleContext, compressedApkActionBuilder);

      singleJarActionBuilder.addInput(compressedApk);
      singleJarCommandLine.addExecPath("--sources", compressedApk);

      addClassesDex(compressedApkActionBuilder, compressedApkCommandLine);
      addJavaResourceFile(compressedApkActionBuilder, compressedApkCommandLine);
      addNativeLibs(compressedApkActionBuilder, compressedApkCommandLine);

      if (javaResourceZip != null) {
        extractedJavaResourceZip = createResourceExtractionAction(ruleContext);
        if (ruleContext.getFragment(AndroidConfiguration.class).compressJavaResources()) {
          compressedApkActionBuilder.addInput(extractedJavaResourceZip);
          compressedApkCommandLine.addExecPath("--sources", extractedJavaResourceZip);
        }
      }

      addNoCompressExtensions(ruleContext, compressedApkCommandLine);
      compressedApkActionBuilder.addCommandLine(compressedApkCommandLine.build());
      ruleContext.registerAction(compressedApkActionBuilder.build(ruleContext));

    } else {

      addClassesDex(singleJarActionBuilder, singleJarCommandLine);
      addJavaResourceFile(singleJarActionBuilder, singleJarCommandLine);
      addNativeLibs(singleJarActionBuilder, singleJarCommandLine);

    }

    if (nativeLibs.getName() != null) {
      singleJarActionBuilder.addInput(nativeLibs.getName());
      singleJarCommandLine
          .add("--resources")
          .addFormatted("%s:%s", nativeLibs.getName(), nativeLibs.getName().getFilename());
    }

    for (Artifact inputZip : inputZips.build()) {
      singleJarActionBuilder.addInput(inputZip);
      singleJarCommandLine.addExecPath("--sources", inputZip);
    }

    addNoCompressExtensions(ruleContext, singleJarCommandLine);

    if (extractedJavaResourceZip != null
        && !ruleContext.getFragment(AndroidConfiguration.class).compressJavaResources()) {
      singleJarActionBuilder.addInput(extractedJavaResourceZip);
      singleJarCommandLine.addExecPath("--sources", extractedJavaResourceZip);
    }

    singleJarActionBuilder.addCommandLine(singleJarCommandLine.build());
    ruleContext.registerAction(singleJarActionBuilder.build(ruleContext));
  }

  private void addNativeLibs(Builder actionBuilder,
      CustomCommandLine.Builder commandLineBuilder) {
    for (String architecture : nativeLibs.getMap().keySet()) {
      for (Artifact nativeLib : nativeLibs.getMap().get(architecture)) {
        actionBuilder.addInput(nativeLib);
        commandLineBuilder
            .add("--resources")
            .addFormatted("%s:lib/%s/%s", nativeLib, architecture, nativeLib.getFilename());
      }
    }
  }

  private void addJavaResourceFile(Builder actionBuilder,
      CustomCommandLine.Builder commandLineBuilder) {
    if (javaResourceFile != null) {
      actionBuilder.addInput(javaResourceFile);
      commandLineBuilder
          .add("--resources")
          .addFormatted("%s:%s", javaResourceFile, javaResourceFile.getFilename());
    }
  }

  private void addClassesDex(Builder actionBuilder, CustomCommandLine.Builder commandLineBuilder) {
    if (classesDex != null) {
      actionBuilder.addInput(classesDex);
      if (classesDex.getFilename().endsWith(".zip")) {
        commandLineBuilder.addExecPath("--sources", classesDex);
      } else {
       commandLineBuilder
            .add("--resources")
            .addFormatted("%s:%s", classesDex, classesDex.getFilename());
      }
    }
  }

  private void addNoCompressExtensions(RuleContext ruleContext,
      CustomCommandLine.Builder commandLineBuilder) {
    List<String> noCompressExtensions;
    if (ruleContext
        .getRule()
        .isAttrDefined(AndroidRuleClasses.NOCOMPRESS_EXTENSIONS_ATTR, Type.STRING_LIST)) {
      noCompressExtensions =
          ruleContext
              .getExpander()
              .withDataLocations()
              .tokenized(AndroidRuleClasses.NOCOMPRESS_EXTENSIONS_ATTR);
    } else {
      // This code is also used by android_test, which doesn't have this attribute.
      noCompressExtensions = ImmutableList.of();
    }
    if (!noCompressExtensions.isEmpty()) {
      commandLineBuilder.addAll("--nocompress_suffixes", noCompressExtensions);
    }
  }

  private Artifact createResourceExtractionAction(RuleContext ruleContext) {
      // The javaResourceZip contains many files that are unwanted in the APK such as .class files.
      Artifact extractedJavaResourceZip =
          getApkArtifact(ruleContext, "extracted_" + javaResourceZip.getFilename());
      ruleContext.registerAction(
          new Builder()
              .setExecutable(resourceExtractor)
              .setMnemonic("ResourceExtractor")
              .setProgressMessage("Extracting Java resources from deploy jar for %s", apkName)
              .addInput(javaResourceZip)
              .addOutput(extractedJavaResourceZip)
              .addCommandLine(
                  CustomCommandLine.builder()
                      .addExecPath(javaResourceZip)
                      .addExecPath(extractedJavaResourceZip)
                      .build())
              .build(ruleContext));

      return extractedJavaResourceZip;
  }

  /** Uses the zipalign tool to align the zip boundaries for uncompressed resources by 4 bytes. */
  private void zipalignApk(RuleContext ruleContext, Artifact inputApk, Artifact zipAlignedApk) {
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInput(inputApk)
            .addOutput(zipAlignedApk)
            .setExecutable(AndroidSdkProvider.fromRuleContext(ruleContext).getZipalign())
            .setProgressMessage("Zipaligning %s", apkName)
            .setMnemonic("AndroidZipAlign")
            .addInput(inputApk)
            .addOutput(zipAlignedApk)
            .addCommandLine(
                CustomCommandLine.builder()
                    .add("-p") // memory page aligment for stored shared object files
                    .add("4")
                    .addExecPath(inputApk)
                    .addExecPath(zipAlignedApk)
                    .build())
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
            .addInput(signingKey)
            .addOutput(signedAndZipalignedApk)
            .addInput(unsignedApk)
            .addCommandLine(
                CustomCommandLine.builder()
                    .add("sign")
                    .add("--ks")
                    .addExecPath(signingKey)
                    .add("--ks-pass", "pass:android")
                    .add("--v1-signing-enabled", Boolean.toString(signingMethod.signV1()))
                    .add("--v1-signer-name", "CERT")
                    .add("--v2-signing-enabled", Boolean.toString(signingMethod.signV2()))
                    .add("--out")
                    .addExecPath(signedAndZipalignedApk)
                    .addExecPath(unsignedApk)
                    .build())
            .build(ruleContext));
  }

  // Adds the appropriate SpawnAction options depending on if SingleJar is a jar or not.
  private static void setSingleJarAsExecutable(
      RuleContext ruleContext, SpawnAction.Builder builder) {
    Artifact singleJar = JavaToolchainProvider.from(ruleContext).getSingleJar();
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .setJarExecutable(
              JavaCommon.getHostJavaExecutable(ruleContext),
              singleJar,
              JavaToolchainProvider.from(ruleContext).getJvmOptions())
          .addTransitiveInputs(JavaRuntimeInfo.forHost(ruleContext).javaBaseInputsMiddleman());
    } else {
      builder.setExecutable(singleJar);
    }
  }

  private Artifact getApkArtifact(RuleContext ruleContext, String baseName) {
    if (artifactLocation != null) {
      return ruleContext.getUniqueDirectoryArtifact(
          artifactLocation, baseName, ruleContext.getBinOrGenfilesDirectory());
    } else {
      return AndroidBinary.getDxArtifact(ruleContext, baseName);
    }
  }
}
