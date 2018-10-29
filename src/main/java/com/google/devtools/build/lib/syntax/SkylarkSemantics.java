// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.syntax;

import com.google.auto.value.AutoValue;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.List;
import java.util.function.Function;

/**
 * Options that affect Skylark semantics.
 *
 * <p>For descriptions of what these options do, see {@link SkylarkSemanticsOptions}.
 */
// TODO(brandjon): User error messages that reference options should maybe be substituted with the
// option name outside of the core Skylark interpreter?
// TODO(brandjon): Eventually these should be documented in full here, and SkylarkSemanticsOptions
// should refer to this class for documentation. But this doesn't play nice with the options
// parser's annotation mechanism.
@AutoValue
public abstract class SkylarkSemantics {

  /**
   * Enum where each element represents a skylark semantics flag. The name of each value should
   * be the exact name of the flag transformed to upper case (for error representation).
   */
  public enum FlagIdentifier {
    EXPERIMENTAL_ANALYSIS_TESTING_IMPROVEMENTS(
        SkylarkSemantics::experimentalAnalysisTestingImprovements),
    EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS(
        SkylarkSemantics::experimentalEnableAndroidMigrationApis),
    EXPERIMENTAL_PLATFORM_API(SkylarkSemantics::experimentalPlatformsApi),
    INCOMPATIBLE_DISABLE_OBJC_PROVIDER_RESOURCES(
        SkylarkSemantics::incompatibleDisableObjcProviderResources),
    INCOMPATIBLE_NO_TARGET_OUTPUT_GROUP(
        SkylarkSemantics::incompatibleNoTargetOutputGroup),
    INCOMPATIBLE_NO_ATTR_LICENSE(SkylarkSemantics::incompatibleNoAttrLicense),
    NONE(null);

    // Using a Function here makes the enum definitions far cleaner, and, since this is
    // a private field, and we can ensure no callers treat this field as mutable.
    @SuppressWarnings("ImmutableEnumChecker")
    private final Function<SkylarkSemantics, Boolean> semanticsFunction;

    FlagIdentifier(Function<SkylarkSemantics, Boolean> semanticsFunction) {
      this.semanticsFunction = semanticsFunction;
    }

    /**
     * Returns the name of the flag that this identifier controls. For example, EXPERIMENTAL_FOO
     * would return 'experimental_foo'.
     */
    public String getFlagName() {
      return Ascii.toLowerCase(this.name());
    }
  }

  /**
   * Returns true if a feature attached to the given toggling flags should be enabled.
   *
   * <ul>
   *   <li>If both parameters are {@code NONE}, this indicates the feature is not
   *       controlled by flags, and should thus be enabled.</li>
   *   <li>If the {@code enablingFlag} parameter is non-{@code NONE}, this returns
   *       true if and only if that flag is true. (This represents a feature that is only on
   *       if a given flag is *on*).</li>
   *   <li>If the {@code disablingFlag} parameter is non-{@code NONE}, this returns
   *       true if and only if that flag is false. (This represents a feature that is only on
   *       if a given flag is *off*).</li>
   *   <li>It is illegal to pass both parameters as non-{@code NONE}.</li>
   * </ul>
   */
  public boolean isFeatureEnabledBasedOnTogglingFlags(
      FlagIdentifier enablingFlag,
      FlagIdentifier disablingFlag) {
    Preconditions.checkArgument(enablingFlag == FlagIdentifier.NONE
        || disablingFlag == FlagIdentifier.NONE,
        "at least one of 'enablingFlag' or 'disablingFlag' must be NONE");
    if (enablingFlag != FlagIdentifier.NONE) {
      return enablingFlag.semanticsFunction.apply(this);
    } else {
      return disablingFlag == FlagIdentifier.NONE || !disablingFlag.semanticsFunction.apply(this);
    }
  }

  /**
   * The AutoValue-generated concrete class implementing this one.
   *
   * <p>AutoValue implementation classes are usually package-private. We expose it here for the
   * benefit of code that relies on reflection.
   */
  public static final Class<? extends SkylarkSemantics> IMPL_CLASS =
      AutoValue_SkylarkSemantics.class;

  // <== Add new options here in alphabetic order ==>
  public abstract boolean experimentalAnalysisTestingImprovements();

  public abstract List<String> experimentalCcSkylarkApiEnabledPackages();

  public abstract boolean experimentalEnableAndroidMigrationApis();

  public abstract boolean experimentalEnableRepoMapping();

  public abstract boolean experimentalRemapMainRepo();

  public abstract boolean experimentalPlatformsApi();

  public abstract boolean experimentalStarlarkConfigTransitions();

  public abstract boolean incompatibleBzlDisallowLoadAfterStatement();

  public abstract boolean incompatibleDepsetIsNotIterable();

  public abstract boolean incompatibleDepsetUnion();

  public abstract boolean incompatibleDisableDeprecatedAttrParams();

  public abstract boolean incompatibleDisableObjcProviderResources();

  public abstract boolean incompatibleDisallowConflictingProviders();

  public abstract boolean incompatibleDisallowDataTransition();

  public abstract boolean incompatibleDisallowDictPlus();

  public abstract boolean incompatibleDisallowFileType();

  public abstract boolean incompatibleDisallowLegacyJavaInfo();

  public abstract boolean incompatibleDisallowLoadLabelsToCrossPackageBoundaries();

  public abstract boolean incompatibleDisallowOldStyleArgsAdd();

  public abstract boolean incompatibleDisallowSlashOperator();

  public abstract boolean incompatibleExpandDirectories();

  public abstract boolean incompatibleGenerateJavaCommonSourceJar();

  public abstract boolean incompatibleNewActionsApi();

  public abstract boolean incompatibleNoAttrLicense();

  public abstract boolean incompatibleNoOutputAttrDefault();

  public abstract boolean incompatibleNoSupportToolsInActionInputs();

  public abstract boolean incompatibleNoTargetOutputGroup();

  public abstract boolean incompatibleNoTransitiveLoads();

  public abstract boolean incompatiblePackageNameIsAFunction();

  public abstract boolean incompatibleRangeType();

  public abstract boolean incompatibleRemoveNativeGitRepository();

  public abstract boolean incompatibleRemoveNativeHttpArchive();

  public abstract boolean incompatibleStaticNameResolution();

  public abstract boolean incompatibleStringIsNotIterable();

  public abstract boolean internalSkylarkFlagTestCanary();

  /** Returns a {@link Builder} initialized with the values of this instance. */
  public abstract Builder toBuilder();

  public static Builder builder() {
    return new AutoValue_SkylarkSemantics.Builder();
  }

  /** Returns a {@link Builder} initialized with default values for all options. */
  public static Builder builderWithDefaults() {
    return DEFAULT_SEMANTICS.toBuilder();
  }

  public static final SkylarkSemantics DEFAULT_SEMANTICS =
      builder()
          // <== Add new options here in alphabetic order ==>
          .experimentalAnalysisTestingImprovements(false)
          .experimentalCcSkylarkApiEnabledPackages(ImmutableList.of())
          .experimentalEnableAndroidMigrationApis(false)
          .experimentalEnableRepoMapping(false)
          .experimentalRemapMainRepo(false)
          .experimentalPlatformsApi(false)
          .experimentalStarlarkConfigTransitions(false)
          .incompatibleBzlDisallowLoadAfterStatement(false)
          .incompatibleDepsetIsNotIterable(false)
          .incompatibleDepsetUnion(false)
          .incompatibleDisableDeprecatedAttrParams(false)
          .incompatibleDisableObjcProviderResources(false)
          .incompatibleDisallowConflictingProviders(true)
          .incompatibleDisallowDataTransition(false)
          .incompatibleDisallowDictPlus(false)
          .incompatibleDisallowFileType(false)
          .incompatibleDisallowLegacyJavaInfo(false)
          .incompatibleDisallowLoadLabelsToCrossPackageBoundaries(false)
          .incompatibleDisallowOldStyleArgsAdd(false)
          .incompatibleDisallowSlashOperator(false)
          .incompatibleExpandDirectories(false)
          .incompatibleGenerateJavaCommonSourceJar(false)
          .incompatibleNewActionsApi(false)
          .incompatibleNoAttrLicense(false)
          .incompatibleNoOutputAttrDefault(false)
          .incompatibleNoSupportToolsInActionInputs(false)
          .incompatibleNoTargetOutputGroup(false)
          .incompatibleNoTransitiveLoads(false)
          .incompatiblePackageNameIsAFunction(false)
          .incompatibleRangeType(true)
          .incompatibleRemoveNativeGitRepository(true)
          .incompatibleRemoveNativeHttpArchive(true)
          .incompatibleStaticNameResolution(false)
          .incompatibleStringIsNotIterable(false)
          .internalSkylarkFlagTestCanary(false)
          .build();

  /** Builder for {@link SkylarkSemantics}. All fields are mandatory. */
  @AutoValue.Builder
  public abstract static class Builder {

    // <== Add new options here in alphabetic order ==>
    public abstract Builder experimentalAnalysisTestingImprovements(boolean value);

    public abstract Builder experimentalCcSkylarkApiEnabledPackages(List<String> value);

    public abstract Builder experimentalEnableAndroidMigrationApis(boolean value);

    public abstract Builder experimentalEnableRepoMapping(boolean value);

    public abstract Builder experimentalRemapMainRepo(boolean value);

    public abstract Builder experimentalPlatformsApi(boolean value);

    public abstract Builder experimentalStarlarkConfigTransitions(boolean value);

    public abstract Builder incompatibleBzlDisallowLoadAfterStatement(boolean value);

    public abstract Builder incompatibleDepsetIsNotIterable(boolean value);

    public abstract Builder incompatibleDepsetUnion(boolean value);

    public abstract Builder incompatibleDisableDeprecatedAttrParams(boolean value);

    public abstract Builder incompatibleDisableObjcProviderResources(boolean value);

    public abstract Builder incompatibleDisallowConflictingProviders(boolean value);

    public abstract Builder incompatibleDisallowDataTransition(boolean value);

    public abstract Builder incompatibleDisallowDictPlus(boolean value);

    public abstract Builder incompatibleDisallowFileType(boolean value);

    public abstract Builder incompatibleDisallowLegacyJavaInfo(boolean value);

    public abstract Builder incompatibleDisallowLoadLabelsToCrossPackageBoundaries(boolean value);

    public abstract Builder incompatibleDisallowOldStyleArgsAdd(boolean value);

    public abstract Builder incompatibleDisallowSlashOperator(boolean value);

    public abstract Builder incompatibleExpandDirectories(boolean value);

    public abstract Builder incompatibleGenerateJavaCommonSourceJar(boolean value);

    public abstract Builder incompatibleNewActionsApi(boolean value);

    public abstract Builder incompatibleNoAttrLicense(boolean value);

    public abstract Builder incompatibleNoOutputAttrDefault(boolean value);

    public abstract Builder incompatibleNoSupportToolsInActionInputs(boolean value);

    public abstract Builder incompatibleNoTargetOutputGroup(boolean value);

    public abstract Builder incompatibleNoTransitiveLoads(boolean value);

    public abstract Builder incompatiblePackageNameIsAFunction(boolean value);

    public abstract Builder incompatibleRangeType(boolean value);

    public abstract Builder incompatibleRemoveNativeGitRepository(boolean value);

    public abstract Builder incompatibleRemoveNativeHttpArchive(boolean value);

    public abstract Builder incompatibleStaticNameResolution(boolean value);

    public abstract Builder incompatibleStringIsNotIterable(boolean value);

    public abstract Builder internalSkylarkFlagTestCanary(boolean value);

    public abstract SkylarkSemantics build();
  }
}
