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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MERGE_ZIP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STORYBOARD;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STRINGS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_ASSET_CATALOGS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_DEFAULT_PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_ENTITLEMENTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_ICON_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_INFOPLISTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_RESOURCES_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_STORYBOARDS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_STRINGS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchApplicationBundleRule.WATCH_APP_STRUCTURED_RESOURCES_ATTR;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;
import com.google.devtools.build.lib.rules.objc.WatchUtils.WatchOSVersion;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.xcode.xcodegen.proto.XcodeGenProtos.XcodeprojBuildSetting;
import javax.annotation.Nullable;

/**
 * Contains support methods to build watch application bundle - does normal bundle processing -
 * resources, plists and creates a final (signed if necessary) bundle.
 */
final class WatchApplicationSupport {

  private final RuleContext ruleContext;
  private final WatchOSVersion watchOSVersion;
  private final ImmutableSet<Attribute> dependencyAttributes;
  private final String bundleName;
  private final IntermediateArtifacts intermediateArtifacts;
  private final Attributes attributes;
  private final Artifact ipaArtifact;
  private final String artifactPrefix;

  /**
   * @param ruleContext the current rule context
   * @param watchOSVersion the version of watchOS for which to create an application bundle
   * @param dependencyAttributes attributes on the current rule context to obtain transitive
   *     resources from
   * @param intermediateArtifacts the utility object to obtain namespacing for intermediate bundling
   *     artifacts
   * @param bundleName the name of the bundle
   * @param ipaArtifact the output ipa created by this application bundling
   * @param artifactPrefix the string prefix to prepend to bundling artifacts for the application --
   *     this prevents intermediate artifacts under this same rule context (such as watch extension
   *     bundling) from conflicting
   */
  WatchApplicationSupport(
      RuleContext ruleContext,
      WatchOSVersion watchOSVersion,
      ImmutableSet<Attribute> dependencyAttributes,
      IntermediateArtifacts intermediateArtifacts,
      String bundleName,
      Artifact ipaArtifact,
      String artifactPrefix) {
    this.ruleContext = ruleContext;
    this.watchOSVersion = watchOSVersion;
    this.dependencyAttributes = dependencyAttributes;
    this.intermediateArtifacts = intermediateArtifacts;
    this.bundleName = bundleName;
    this.ipaArtifact = ipaArtifact;
    this.artifactPrefix = artifactPrefix;
    this.attributes = new Attributes(ruleContext);
  }

  /**
   * Registers actions to create a watch application bundle.
   *
   * @param innerBundleZips any zip files to be unzipped and merged into the application bundle
   * @param filesToBuild files to build for the rule; the watchOS application .ipa is added to this
   *     set
   * @param exposedObjcProviderBuilder provider builder which watch application bundle outputs are
   *     added to (for later consumption by depending rules)
   */
  void createBundle(
      Iterable<Artifact> innerBundleZips,
      NestedSetBuilder<Artifact> filesToBuild,
      ObjcProvider.Builder exposedObjcProviderBuilder)
      throws InterruptedException {

    ObjcProvider objcProvider = objcProvider(innerBundleZips);

    createBundle(
        Optional.<XcodeProvider.Builder>absent(),
        objcProvider,
        filesToBuild,
        exposedObjcProviderBuilder);
  }

  /**
   * Registers actions to create a watch application bundle and xcode project.
   *
   * @param xcodeProviderBuilder provider builder which xcode project generation information is
   *     added to (for later consumption by depending rules)
   * @param innerBundleZips any zip files to be unzipped and merged into the application bundle
   * @param filesToBuild files to build for the rule; the watchOS application .ipa is added to this
   *     set
   * @param exposedObjcProviderBuilder provider builder which watch application bundle outputs are
   *     added to (for later consumption by depending rules)
   */
  void createBundleAndXcodeproj(
      XcodeProvider.Builder xcodeProviderBuilder,
      Iterable<Artifact> innerBundleZips,
      NestedSetBuilder<Artifact> filesToBuild,
      ObjcProvider.Builder exposedObjcProviderBuilder)
      throws InterruptedException {
    ObjcProvider objcProvider = objcProvider(innerBundleZips);

    createBundle(
        Optional.of(xcodeProviderBuilder), objcProvider, filesToBuild, exposedObjcProviderBuilder);

    // Add common watch settings.
    WatchUtils.addXcodeSettings(ruleContext, xcodeProviderBuilder);

    // Add watch application specific xcode settings.
    addXcodeSettings(xcodeProviderBuilder);

    XcodeSupport xcodeSupport =
        new XcodeSupport(ruleContext, intermediateArtifacts, labelForWatchApplication())
            .addXcodeSettings(
                xcodeProviderBuilder,
                objcProvider,
                watchOSVersion.getApplicationXcodeProductType(),
                ruleContext.getFragment(AppleConfiguration.class).getIosCpu(),
                ConfigurationDistinguisher.WATCH_OS1_EXTENSION);

    for (Attribute attribute : dependencyAttributes) {
      xcodeSupport.addDependencies(xcodeProviderBuilder, attribute);
    }
  }

  private void createBundle(
      Optional<XcodeProvider.Builder> xcodeProviderBuilder,
      ObjcProvider depsObjcProvider,
      NestedSetBuilder<Artifact> filesToBuild,
      ObjcProvider.Builder exposedObjcProviderBuilder)
      throws InterruptedException {
    registerActions();

    ReleaseBundling.Builder releaseBundling = new ReleaseBundling.Builder()
        .setIpaArtifact(ipaArtifact)
        .setBundleId(attributes.bundleId())
        .setAppIcon(attributes.appIcon())
        .setProvisioningProfile(attributes.provisioningProfile())
        .setProvisioningProfileAttributeName(WATCH_APP_PROVISIONING_PROFILE_ATTR)
        .setTargetDeviceFamilies(families())
        .setIntermediateArtifacts(intermediateArtifacts)
        .setInfoPlistsFromRule(attributes.infoPlists())
        .setArtifactPrefix(artifactPrefix)
        .setEntitlements(attributes.entitlements());

    if (attributes.isBundleIdExplicitySpecified()) {
      releaseBundling.setPrimaryBundleId(attributes.bundleId());
    } else {
      releaseBundling.setFallbackBundleId(attributes.bundleId());
    }

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    PlatformType appPlatformType = watchOSVersion == WatchOSVersion.OS1
         ? PlatformType.IOS : PlatformType.WATCHOS;
    ReleaseBundlingSupport releaseBundlingSupport =
        new ReleaseBundlingSupport(
                ruleContext,
                depsObjcProvider,
                LinkedBinary.DEPENDENCIES_ONLY,
                watchOSVersion.getApplicationBundleDirFormat(),
                bundleName,
                WatchUtils.determineMinimumOsVersion(
                    ObjcRuleClasses.objcConfiguration(ruleContext).getMinimumOs()),
                releaseBundling.build(),
                appleConfiguration.getMultiArchPlatform(appPlatformType))
            .registerActions(DsymOutputType.APP);

    if (xcodeProviderBuilder.isPresent()) {
      releaseBundlingSupport.addXcodeSettings(xcodeProviderBuilder.get());
    }

    releaseBundlingSupport
        .addFilesToBuild(filesToBuild, DsymOutputType.APP)
        .validateResources()
        .validateAttributes()
        .addExportedDebugArtifacts(exposedObjcProviderBuilder, DsymOutputType.APP);
  }

  /**
   * Returns the {@link TargetDeviceFamily} that the watch application bundle is targeting.
   * For simulator builds, this returns a set of {@code TargetDeviceFamily.IPHONE} and
   * {@code TargetDeviceFamily.WATCH} and for non-simulator builds, this returns
   * {@code TargetDeviceFamily.WATCH}.
   */
  private ImmutableSet<TargetDeviceFamily> families() {
    Platform platform =
        ruleContext.getFragment(AppleConfiguration.class).getMultiArchPlatform(PlatformType.IOS);
    if (platform == Platform.IOS_DEVICE) {
      return ImmutableSet.of(TargetDeviceFamily.WATCH);
    } else {
      return ImmutableSet.of(TargetDeviceFamily.IPHONE, TargetDeviceFamily.WATCH);
    }
  }

  /**
   * Adds watch application specific xcode settings - TARGETED_DEVICE_FAMILY is set to "1, 4"
   * for enabling building for simulator.
   */
  private void addXcodeSettings(XcodeProvider.Builder xcodeProviderBuilder) {
    xcodeProviderBuilder.addMainTargetXcodeprojBuildSettings(ImmutableList.of(
        XcodeprojBuildSetting.newBuilder()
        .setName("TARGETED_DEVICE_FAMILY[sdk=iphonesimulator*]")
        .setValue(Joiner.on(',').join(TargetDeviceFamily.UI_DEVICE_FAMILY_VALUES.get(
            families())))
        .build()));
  }

  /**
   * Registers actions to copy WatchKit stub binary at
   * $(SDK_ROOT)/Library/Application Support/WatchKit/WK as bundle binary and as stub resource.
   *
   * For example, for a bundle named "Foo.app", the contents will be,
   *    - Foo.app/Foo (WK stub as binary)
   *    - Foo.app/_WatchKitStub/WK  (WK stub as resource)
   */
  private void registerActions() {
    Artifact watchKitStubZip = watchKitStubZip();
    String workingDirectory = watchKitStubZip.getExecPathString()
        .substring(0, watchKitStubZip.getExecPathString().lastIndexOf('/'));
    String watchKitStubBinaryPath = workingDirectory + "/" + bundleName;
    String watchKitStubResourcePath = workingDirectory + "/_WatchKitStub";

    ImmutableList<String> command = ImmutableList.of(
        // 1. Copy WK stub as binary
        String.format("cp -f %s %s", WatchUtils.WATCH_KIT_STUB_PATH, watchKitStubBinaryPath),
        "&&",
        // 2. Copy WK stub as bundle resource.
        "mkdir -p " + watchKitStubResourcePath,
        "&&",
        String.format("cp -f %s %s", WatchUtils.WATCH_KIT_STUB_PATH, watchKitStubResourcePath),
         // 3. Zip them.
        "&&",
        "cd " + workingDirectory,
        "&&",
        String.format(
            "/usr/bin/zip -q -r -0 %s %s",
            watchKitStubZip.getFilename(),
            Joiner.on(" ").join(ImmutableList.of("_WatchKitStub", bundleName))));

    ruleContext.registerAction(
        ObjcRuleClasses.spawnAppleEnvActionBuilder(
                ruleContext,
                ruleContext
                    .getFragment(AppleConfiguration.class)
                    .getMultiArchPlatform(PlatformType.WATCHOS))
            .setProgressMessage(
                "Copying WatchKit binary and stub resource: " + ruleContext.getLabel())
            .setShellCommand(ImmutableList.of("/bin/bash", "-c", Joiner.on(" ").join(command)))
            .addOutput(watchKitStubZip)
            .build(ruleContext));
  }

  private ObjcProvider objcProvider(Iterable<Artifact> innerBundleZips) {
    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();
    objcProviderBuilder.addAll(MERGE_ZIP, innerBundleZips);

    // Add all resource files applicable to watch application from dependency providers.
    for (Attribute attribute : dependencyAttributes) {
      Iterable<ObjcProvider> dependencyObjcProviders = ruleContext.getPrerequisites(
          attribute.getName(), attribute.getAccessMode(), ObjcProvider.class);
      for (ObjcProvider dependencyObjcProvider : dependencyObjcProviders) {
        objcProviderBuilder.addTransitiveAndPropagate(GENERAL_RESOURCE_FILE,
            dependencyObjcProvider);
        objcProviderBuilder.addTransitiveAndPropagate(GENERAL_RESOURCE_DIR,
            dependencyObjcProvider);
        objcProviderBuilder.addTransitiveAndPropagate(BUNDLE_FILE,
            dependencyObjcProvider);
        objcProviderBuilder.addTransitiveAndPropagate(XCASSETS_DIR,
            dependencyObjcProvider);
        objcProviderBuilder.addTransitiveAndPropagate(ASSET_CATALOG,
            dependencyObjcProvider);
        objcProviderBuilder.addTransitiveAndPropagate(STRINGS,
            dependencyObjcProvider);
        objcProviderBuilder.addTransitiveAndPropagate(STORYBOARD,
            dependencyObjcProvider);
      }
    }
    // Add zip containing WatchKit stubs.
    objcProviderBuilder.add(ObjcProvider.MERGE_ZIP, watchKitStubZip());

    // Add resource files.
    objcProviderBuilder.addAll(GENERAL_RESOURCE_FILE, attributes.storyboards())
      .addAll(GENERAL_RESOURCE_FILE, attributes.resources())
      .addAll(GENERAL_RESOURCE_FILE, attributes.strings())
      .addAll(GENERAL_RESOURCE_DIR,
          ObjcCommon.xcodeStructuredResourceDirs(attributes.structuredResources()))
      .addAll(BUNDLE_FILE, BundleableFile.flattenedRawResourceFiles(attributes.resources()))
      .addAll(
          BUNDLE_FILE,
          BundleableFile.structuredRawResourceFiles(attributes.structuredResources()))
      .addAll(XCASSETS_DIR, ObjcCommon.uniqueContainers(attributes.assetCatalogs(),
          ObjcCommon.ASSET_CATALOG_CONTAINER_TYPE))
      .addAll(ASSET_CATALOG, attributes.assetCatalogs())
      .addAll(STRINGS, attributes.strings())
      .addAll(STORYBOARD, attributes.storyboards());

    return objcProviderBuilder.build();
  }

  private Label labelForWatchApplication()
      throws InterruptedException {
    try {
      return Label.create(ruleContext.getLabel().getPackageName(), bundleName);
    } catch (LabelSyntaxException labelSyntaxException) {
        throw new InterruptedException("Exception while creating target label for watch "
            + "appplication " + labelSyntaxException);
    }
  }

  /**
   * Returns a zip {@link Artifact} containing stub binary and stub resource that are to be added
   * to the bundle.
   */
  private Artifact watchKitStubZip() {
    return ruleContext.getRelatedArtifact(
        ruleContext.getUniqueDirectory("_watch"), "/WatchKitStub.zip");
  }

  /**
   * Rule attributes used for creating watch application bundle.
   */
  private static class Attributes {
    private final RuleContext ruleContext;

    private Attributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    @Nullable
    String appIcon() {
      return Strings.emptyToNull(ruleContext.attributes().get(WATCH_APP_ICON_ATTR, Type.STRING));
    }

    @Nullable
    Artifact provisioningProfile() {
      Artifact explicitProvisioningProfile =
          getPrerequisiteArtifact(WATCH_APP_PROVISIONING_PROFILE_ATTR);
      if (explicitProvisioningProfile != null) {
        return explicitProvisioningProfile;
      }
      return getPrerequisiteArtifact(WATCH_APP_DEFAULT_PROVISIONING_PROFILE_ATTR);
    }

    String bundleId() {
      Preconditions.checkState(!Strings.isNullOrEmpty(
          ruleContext.attributes().get(WATCH_APP_BUNDLE_ID_ATTR, Type.STRING)),
          "requires a bundle_id value");
      return ruleContext.attributes().get(WATCH_APP_BUNDLE_ID_ATTR, Type.STRING);
    }

    ImmutableList<Artifact> infoPlists() {
      return getPrerequisiteArtifacts(WATCH_APP_INFOPLISTS_ATTR);
    }

    ImmutableList<Artifact> assetCatalogs() {
      return getPrerequisiteArtifacts(WATCH_APP_ASSET_CATALOGS_ATTR);
    }

    ImmutableList<Artifact> strings() {
      return getPrerequisiteArtifacts(WATCH_APP_STRINGS_ATTR);
    }

    ImmutableList<Artifact> storyboards() {
      return getPrerequisiteArtifacts(WATCH_APP_STORYBOARDS_ATTR);
    }

    ImmutableList<Artifact> resources() {
      return getPrerequisiteArtifacts(WATCH_APP_RESOURCES_ATTR);
    }

    ImmutableList<Artifact> structuredResources() {
      return getPrerequisiteArtifacts(WATCH_APP_STRUCTURED_RESOURCES_ATTR);
    }

    @Nullable
    Artifact entitlements() {
      return getPrerequisiteArtifact(WATCH_APP_ENTITLEMENTS_ATTR);
    }

    private boolean isBundleIdExplicitySpecified() {
      return ruleContext.attributes().isAttributeValueExplicitlySpecified(WATCH_APP_BUNDLE_ID_ATTR);
    }

    private ImmutableList<Artifact> getPrerequisiteArtifacts(String attribute) {
      return ruleContext.getPrerequisiteArtifacts(attribute, Mode.TARGET).list();
    }

    @Nullable
    private Artifact getPrerequisiteArtifact(String attribute) {
      return ruleContext.getPrerequisiteArtifact(attribute, Mode.TARGET);
    }
  }
}
