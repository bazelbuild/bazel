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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STRINGS;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.BundlingRule.FAMILIES_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_DEFAULT_PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_ENTITLEMENTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_FAMILIES_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_INFOPLISTS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_RESOURCES_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_STRINGS_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.WatchExtensionBundleRule.WATCH_EXT_STRUCTURED_RESOURCES_ATTR;

import com.dd.plist.NSDictionary;
import com.dd.plist.NSObject;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;
import com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.InvalidFamilyNameException;
import com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.RepeatedFamilyNameException;
import com.google.devtools.build.lib.syntax.Type;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Contains support methods to build watch extension bundle - does normal bundle processing -
 * compiling and linking the binary, resources, plists and creates a final (signed if necessary)
 * bundle.
 */
public class Watch2ExtensionSupport {

  private final RuleContext ruleContext;
  private final IntermediateArtifacts intermediateArtifacts;
  private final String bundleName;
  private final Attributes attributes;

  /**
   * @param ruleContext the current rule context
   * @param intermediateArtifacts the utility object to obtain namespacing for intermediate bundling
   *     artifacts
   * @param bundleName the name of the bundle
   */
  Watch2ExtensionSupport(
      RuleContext ruleContext, IntermediateArtifacts intermediateArtifacts, String bundleName) {
    this.ruleContext = ruleContext;
    this.intermediateArtifacts = intermediateArtifacts;
    this.bundleName = bundleName;
    this.attributes = new Attributes(ruleContext);
  }

  /**
   * Registers actions to create a watchOS2 extension bundle and zip it into an {@code .ipa}.
   *
   * @param ipaArtifact an .ipa artifact containing to extension bundle; this is the output artifact
   *     of the bundling
   */
  void createBundle(Artifact ipaArtifact) throws InterruptedException {
    ObjcProvider.Builder releaseBundlingObjcProviderBuilder = new ObjcProvider.Builder();
    releaseBundlingObjcProviderBuilder.addTransitiveAndPropagate(attributes.binaryDependencies());
    releaseBundlingObjcProviderBuilder
        .addAll(GENERAL_RESOURCE_FILE, attributes.resources())
        .addAll(GENERAL_RESOURCE_FILE, attributes.strings())
        .addAll(
            GENERAL_RESOURCE_DIR,
            ObjcCommon.xcodeStructuredResourceDirs(attributes.structuredResources()))
        .addAll(BUNDLE_FILE, BundleableFile.flattenedRawResourceFiles(attributes.resources()))
        .addAll(
            BUNDLE_FILE,
            BundleableFile.structuredRawResourceFiles(attributes.structuredResources()))
        .addAll(STRINGS, attributes.strings());
    ObjcProvider releaseBundlingObjcProvider = releaseBundlingObjcProviderBuilder.build();

    registerWatchExtensionAutomaticPlistAction();

    ImmutableSet<TargetDeviceFamily> families = attributes.families();

    if (families.isEmpty()) {
      ruleContext.attributeError(FAMILIES_ATTR, ReleaseBundling.INVALID_FAMILIES_ERROR);
    }

    ReleaseBundling.Builder releaseBundling =
        new ReleaseBundling.Builder()
            .setIpaArtifact(ipaArtifact)
            .setBundleId(attributes.bundleId())
            .setProvisioningProfile(attributes.provisioningProfile())
            .setProvisioningProfileAttributeName(WATCH_EXT_PROVISIONING_PROFILE_ATTR)
            .setTargetDeviceFamilies(families)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setInfoPlistsFromRule(attributes.infoPlists())
            .addInfoplistInput(intermediateArtifacts.watchExtensionAutomaticPlist())
            .setEntitlements(attributes.entitlements());

    if (attributes.isBundleIdExplicitySpecified()) {
      releaseBundling.setPrimaryBundleId(attributes.bundleId());
    } else {
      releaseBundling.setFallbackBundleId(attributes.bundleId());
    }

    ReleaseBundlingSupport releaseBundlingSupport =
        new ReleaseBundlingSupport(
            ruleContext,
            releaseBundlingObjcProvider,
            LinkedBinary.DEPENDENCIES_ONLY,
            ReleaseBundlingSupport.EXTENSION_BUNDLE_DIR_FORMAT,
            bundleName,
            WatchUtils.determineMinimumOsVersion(
                ObjcRuleClasses.objcConfiguration(ruleContext).getMinimumOs()),
            releaseBundling.build());

    releaseBundlingSupport
        .registerActions(DsymOutputType.APP)
        .validateResources()
        .validateAttributes();
  }

  /**
   * Registers an action to generate a plist containing entries required for watch extension that
   * should be added to the merged plist.
   */
  private void registerWatchExtensionAutomaticPlistAction() {
    NSDictionary watchExtensionAutomaticEntries = new NSDictionary();
    watchExtensionAutomaticEntries.put(
        "UIRequiredDeviceCapabilities", NSObject.wrap(new String[] {"watch-companion"}));

    ruleContext.registerAction(
        new FileWriteAction(
            ruleContext.getActionOwner(),
            intermediateArtifacts.watchExtensionAutomaticPlist(),
            watchExtensionAutomaticEntries.toGnuStepASCIIPropertyList(),
            /*makeExecutable=*/ false));
  }

  /** Rule attributes used for creating watch application bundle. */
  private static class Attributes {
    private final RuleContext ruleContext;

    private Attributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /**
     * Returns the value of the {@code families} attribute in a form that is more useful than a list
     * of strings. Returns an empty set for any invalid {@code families} attribute value, including
     * an empty list.
     */
    ImmutableSet<TargetDeviceFamily> families() {
      List<String> rawFamilies =
          ruleContext.attributes().get(WATCH_EXT_FAMILIES_ATTR, Type.STRING_LIST);
      try {
        return ImmutableSet.copyOf(TargetDeviceFamily.fromNamesInRule(rawFamilies));
      } catch (InvalidFamilyNameException | RepeatedFamilyNameException e) {
        return ImmutableSet.of();
      }
    }

    @Nullable
    Artifact provisioningProfile() {
      Artifact explicitProvisioningProfile =
          getPrerequisiteArtifact(WATCH_EXT_PROVISIONING_PROFILE_ATTR);
      if (explicitProvisioningProfile != null) {
        return explicitProvisioningProfile;
      }
      return getPrerequisiteArtifact(WATCH_EXT_DEFAULT_PROVISIONING_PROFILE_ATTR);
    }

    String bundleId() {
      Preconditions.checkState(
          !Strings.isNullOrEmpty(
              ruleContext.attributes().get(WATCH_EXT_BUNDLE_ID_ATTR, Type.STRING)),
          "requires a bundle_id value");
      return ruleContext.attributes().get(WATCH_EXT_BUNDLE_ID_ATTR, Type.STRING);
    }

    ImmutableList<Artifact> infoPlists() {
      return getPrerequisiteArtifacts(WATCH_EXT_INFOPLISTS_ATTR);
    }

    ImmutableList<Artifact> strings() {
      return getPrerequisiteArtifacts(WATCH_EXT_STRINGS_ATTR);
    }

    ImmutableList<Artifact> resources() {
      return getPrerequisiteArtifacts(WATCH_EXT_RESOURCES_ATTR);
    }

    ImmutableList<Artifact> structuredResources() {
      return getPrerequisiteArtifacts(WATCH_EXT_STRUCTURED_RESOURCES_ATTR);
    }

    Iterable<ObjcProvider> binaryDependencies() {
      return ruleContext.getPrerequisites("binary", Mode.TARGET, ObjcProvider.class);
    }

    @Nullable
    Artifact entitlements() {
      return getPrerequisiteArtifact(WATCH_EXT_ENTITLEMENTS_ATTR);
    }

    private boolean isBundleIdExplicitySpecified() {
      return ruleContext.attributes().isAttributeValueExplicitlySpecified(WATCH_EXT_BUNDLE_ID_ATTR);
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
