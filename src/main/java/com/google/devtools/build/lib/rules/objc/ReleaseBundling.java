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

import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.BundlingRule.FAMILIES_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.APP_ICON_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.BUNDLE_ID_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.DEFAULT_PROVISIONING_PROFILE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.LAUNCH_IMAGE_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.LAUNCH_STORYBOARD_ATTR;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ReleaseBundlingRule.PROVISIONING_PROFILE_ATTR;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.InvalidFamilyNameException;
import com.google.devtools.build.lib.rules.objc.TargetDeviceFamily.RepeatedFamilyNameException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.List;

/**
 * Contains information regarding the creation of a released bundle such as an application
 * or extension. The information which generally includes app icons, launch image, targeted devices
 * and other data for potential signing is used to create a releasable bundle out of the bundle
 * created using {@link Bundling} object.
 */
@Immutable
final class ReleaseBundling {
  static final class Builder {
    private Artifact ipaArtifact;
    private String bundleId;
    private String primaryBundleId;
    private String fallbackBundleId;
    private String appIcon;
    private String launchImage;
    private Artifact launchStoryboard;
    private Artifact provisioningProfile;
    private String provisioningProfileAttributeName;
    private final NestedSetBuilder<Artifact> infoplistInputs = NestedSetBuilder.stableOrder();
    private Iterable<Artifact> infoPlistsFromRule;
    private ImmutableSet<TargetDeviceFamily> families;
    private IntermediateArtifacts intermediateArtifacts;
    private String artifactPrefix;

    public Builder setIpaArtifact(Artifact ipaArtifact) {
      this.ipaArtifact = ipaArtifact;
      return this;
    }

    public Builder setBundleId(String bundleId) {
      this.bundleId = bundleId;
      return this;
    }

    public Builder setPrimaryBundleId(String primaryId) {
      this.primaryBundleId = primaryId;
      return this;
    }

    public Builder setFallbackBundleId(String fallbackId) {
      this.fallbackBundleId = fallbackId;
      return this;
    }

    public Builder setAppIcon(String appIcon) {
      this.appIcon = appIcon;
      return this;
    }

    public Builder setLaunchImage(String launchImage) {
      this.launchImage = launchImage;
      return this;
    }

    public Builder setLaunchStoryboard(Artifact launchStoryboard) {
      this.launchStoryboard = launchStoryboard;
      return this;
    }

    public Builder setProvisioningProfile(Artifact provisioningProfile) {
      this.provisioningProfile = provisioningProfile;
      return this;
    }

    public Builder setProvisioningProfileAttributeName(String provisioningProfileAttributeName) {
      this.provisioningProfileAttributeName = provisioningProfileAttributeName;
      return this;
    }

    public Builder addInfoplistInput(Artifact infoPlist) {
      this.infoplistInputs.add(infoPlist);
      return this;
    }

    public Builder addInfoplistInputs(Iterable<Artifact> infoplists) {
      this.infoplistInputs.addAll(infoplists);
      return this;
    }

    public Builder setInfoPlistsFromRule(Iterable<Artifact> infoPlistsFromRule) {
      this.infoPlistsFromRule = infoPlistsFromRule;
      return this;
    }

    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    public Builder setTargetDeviceFamilies(ImmutableSet<TargetDeviceFamily> families) {
      this.families = families;
      return this;
    }

    public Builder setArtifactPrefix(String artifactPrefix) {
      this.artifactPrefix = artifactPrefix;
      return this;
    }

    public ReleaseBundling build() {
      Preconditions.checkNotNull(intermediateArtifacts, "intermediateArtifacts");
      Preconditions.checkNotNull(families, FAMILIES_ATTR);
      return new ReleaseBundling(
          ipaArtifact,
          bundleId,
          primaryBundleId,
          fallbackBundleId,
          appIcon,
          launchImage,
          launchStoryboard,
          provisioningProfile,
          provisioningProfileAttributeName,
          infoplistInputs.build(),
          infoPlistsFromRule,
          families,
          intermediateArtifacts,
          artifactPrefix);
    }
  }

  /**
   * Returns a {@link ReleaseBundling} object constructed using the information available in given
   * context.
   */
  public static ReleaseBundling releaseBundling(RuleContext ruleContext)
      throws InterruptedException {
    Preconditions.checkState(!Strings.isNullOrEmpty(
        ruleContext.attributes().get(BUNDLE_ID_ATTR, Type.STRING)),
        "requires a bundle_id value");
    String primaryBundleId = null;
    String fallbackBundleId = null;
    Artifact provisioningProfile;

    if (ruleContext.attributes().isAttributeValueExplicitlySpecified(BUNDLE_ID_ATTR)) {
      primaryBundleId = ruleContext.attributes().get(BUNDLE_ID_ATTR, Type.STRING);
    } else {
      fallbackBundleId = ruleContext.attributes().get(BUNDLE_ID_ATTR, Type.STRING);
    }

    Artifact explicitProvisioningProfile =
        ruleContext.getPrerequisiteArtifact(PROVISIONING_PROFILE_ATTR, Mode.TARGET);
    if (explicitProvisioningProfile != null) {
      provisioningProfile = explicitProvisioningProfile;
    } else {
      provisioningProfile = ruleContext.getPrerequisiteArtifact(DEFAULT_PROVISIONING_PROFILE_ATTR,
          Mode.TARGET);
    }

    ImmutableSet<TargetDeviceFamily> families = null;
    List<String> rawFamilies = ruleContext.attributes().get(FAMILIES_ATTR, Type.STRING_LIST);
    try {
      families = ImmutableSet.copyOf(TargetDeviceFamily.fromNamesInRule(rawFamilies));
    } catch (InvalidFamilyNameException | RepeatedFamilyNameException e) {
      families = ImmutableSet.of();
    }

    if (families.isEmpty()) {
      ruleContext.attributeError(FAMILIES_ATTR, INVALID_FAMILIES_ERROR);
    }

    return new ReleaseBundling.Builder()
        .setIpaArtifact(ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA))
        .setBundleId(ruleContext.attributes().get(BUNDLE_ID_ATTR, Type.STRING))
        .setPrimaryBundleId(primaryBundleId)
        .setFallbackBundleId(fallbackBundleId)
        .setAppIcon(Strings.emptyToNull(ruleContext.attributes().get(APP_ICON_ATTR, Type.STRING)))
        .setLaunchImage(Strings.emptyToNull(
            ruleContext.attributes().get(LAUNCH_IMAGE_ATTR, Type.STRING)))
        .setLaunchStoryboard(
            ruleContext.getPrerequisiteArtifact(LAUNCH_STORYBOARD_ATTR, Mode.TARGET))
        .setProvisioningProfile(provisioningProfile)
        .setProvisioningProfileAttributeName(PROVISIONING_PROFILE_ATTR)
        .setTargetDeviceFamilies(families)
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .build();
  }

  @VisibleForTesting
  static final String INVALID_FAMILIES_ERROR =
      "Expected one or two strings from the list 'iphone', 'ipad'";
  private final Artifact ipaArtifact;
  private final String bundleId;
  private final String fallbackBundleId;
  private final String primaryBundleId;
  private final String appIcon;
  private final String launchImage;
  private final Artifact launchStoryboard;
  private final Artifact provisioningProfile;
  private final String provisioningProfileAttributeName;
  private final NestedSet<Artifact> infoplistInputs;
  private final ImmutableSet<TargetDeviceFamily> families;
  private final IntermediateArtifacts intermediateArtifacts;
  private final Iterable<Artifact> infoPlistsFromRule;
  private final String artifactPrefix;

  private ReleaseBundling(
      Artifact ipaArtifact,
      String bundleId,
      String primaryBundleId,
      String fallbackBundleId,
      String appIcon,
      String launchImage,
      Artifact launchStoryboard,
      Artifact provisioningProfile,
      String provisioningProfileAttributeName,
      NestedSet<Artifact> infoplistInputs,
      Iterable<Artifact> infoPlistsFromRule,
      ImmutableSet<TargetDeviceFamily> families,
      IntermediateArtifacts intermediateArtifacts,
      String artifactPrefix) {
    this.ipaArtifact = Preconditions.checkNotNull(ipaArtifact);
    this.bundleId = bundleId;
    this.primaryBundleId = primaryBundleId;
    this.fallbackBundleId = fallbackBundleId;
    this.appIcon = appIcon;
    this.launchImage = launchImage;
    this.launchStoryboard = launchStoryboard;
    this.provisioningProfile = provisioningProfile;
    this.provisioningProfileAttributeName =
        Preconditions.checkNotNull(provisioningProfileAttributeName);
    this.infoplistInputs = Preconditions.checkNotNull(infoplistInputs);
    this.infoPlistsFromRule = infoPlistsFromRule;
    this.families = Preconditions.checkNotNull(families);
    this.intermediateArtifacts = Preconditions.checkNotNull(intermediateArtifacts);
    this.artifactPrefix = artifactPrefix;
  }

  /**
   * Returns the {@link Artifact} containing the final ipa bundle.
   */
  public Artifact getIpaArtifact() {
    return ipaArtifact;
  }

  /**
   * Returns the identifier of this bundle.
   */
  public String getBundleId() {
    return bundleId;
  }

  /**
   * Returns primary bundle ID to use, can be null.
   */
  public String getPrimaryBundleId() {
    return primaryBundleId;
  }

  /**
   * Returns fallback bundle ID to use when primary isn't set.
   */
  public String getFallbackBundleId() {
    return fallbackBundleId;
  }

  /**
   * Returns the app icon name for this bundle, can be null.
   */
  public String getAppIcon() {
    return appIcon;
  }

  /**
   * Returns the launch image name for this bundle, can be null.
   */
  public String getLaunchImage() {
    return launchImage;
  }

  /**
   * Returns an {@link Artifact} containing launch storyboard for this bundle, can be null.
   */
  public Artifact getLaunchStoryboard() {
    return launchStoryboard;
  }

  /**
   * Returns an {@link Artifact} containing provisioning profile used to sign this bundle,
   * can be null.
   */
  public Artifact getProvisioningProfile() {
    return provisioningProfile;
  }

  /**
   * Returns the list of plists to be merged to final bundle.
   */
  public NestedSet<Artifact> getInfoplistInputs() {
    return infoplistInputs;
  }

  /**
   * Returns the list of {@link TargetDeviceFamily} values this bundle is targeting.
   * If empty, the default values specified by {@link FAMILIES_ATTR} will be used.
   */
  public ImmutableSet<TargetDeviceFamily> getTargetDeviceFamilies() {
    return families;
  }

  /**
   * Returns {@link IntermediateArtifacts} used to create this bundle.
   */
  public IntermediateArtifacts getIntermediateArtifacts() {
    return intermediateArtifacts;
  }

  /**
   * Returns the name of the attribute which is used to specifiy the provisioning profile.
   */
  public String getProvisioningProfileAttrName() {
    return provisioningProfileAttributeName;
  }

  /**
   * Adds any info plists specified in the given rule's {@code infoplists} attribute as inputs to
   * this bundle's {@code Info.plist} (which is merged from any such added plists plus some
   * additional information).
   */
  public Iterable<Artifact> getInfoPlistsFromRule() {
    return infoPlistsFromRule;
  }

  /**
   * Returns the prefix to be added to all generated artifact names, can be null. This is useful
   * to disambiguate artifacts for multiple bundles created with different names withing same rule. 
   */
  public String getArtifactPrefix() {
    return artifactPrefix;
  }
}
