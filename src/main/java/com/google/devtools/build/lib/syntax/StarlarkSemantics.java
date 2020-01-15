// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.syntax;

import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.List;
import java.util.function.Function;

/**
 * Options that affect Starlark semantics.
 *
 * <p>For descriptions of what these options do, see {@link StarlarkSemanticsOptions}.
 */
// TODO(brandjon): User error messages that reference options should maybe be substituted with the
// option name outside of the core Starlark interpreter?
// TODO(brandjon): Eventually these should be documented in full here, and StarlarkSemanticsOptions
// should refer to this class for documentation. But this doesn't play nice with the options
// parser's annotation mechanism.
@AutoValue
public abstract class StarlarkSemantics {

  /**
   * Enum where each element represents a starlark semantics flag. The name of each value should be
   * the exact name of the flag transformed to upper case (for error representation).
   */
  // TODO(adonovan): This class, being part of the core Starlark frontend, shouldn't refer to Bazel
  // features. There's no need for an enumeration to represent a set of boolean features. Instead,
  // have StarlarkSemantics hold a set of enabled features (strings), and have callers query
  // features by name. The features can be named string constants, defined close to the code they
  // affect, to avoid accidential misspellings.
  public enum FlagIdentifier {
    EXPERIMENTAL_ACTION_ARGS(StarlarkSemantics::experimentalActionArgs),
    EXPERIMENTAL_ALLOW_INCREMENTAL_REPOSITORY_UPDATES(
        StarlarkSemantics::experimentalAllowIncrementalRepositoryUpdates),
    EXPERIMENTAL_ASPECT_OUTPUT_PROPAGATION(StarlarkSemantics::experimentalAspectOutputPropagation),
    EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS(
        StarlarkSemantics::experimentalEnableAndroidMigrationApis),
    EXPERIMENTAL_BUILD_SETTING_API(StarlarkSemantics::experimentalBuildSettingApi),
    EXPERIMENTAL_GOOGLE_LEGACY_API(StarlarkSemantics::experimentalGoogleLegacyApi),
    EXPERIMENTAL_NINJA_ACTIONS(StarlarkSemantics::experimentalNinjaActions),
    EXPERIMENTAL_PLATFORM_API(StarlarkSemantics::experimentalPlatformsApi),
    EXPERIMENTAL_STARLARK_CONFIG_TRANSITION(
        StarlarkSemantics::experimentalStarlarkConfigTransitions),
    EXPERIMENTAL_STARLARK_UNUSED_INPUTS_LIST(
        StarlarkSemantics::experimentalStarlarkUnusedInputsList),
    EXPERIMENTAL_CC_SHARED_LIBRARY(StarlarkSemantics::experimentalCcSharedLibrary),
    EXPERIMENTAL_REPO_REMOTE_EXEC(StarlarkSemantics::experimentalRepoRemoteExec),
    INCOMPATIBLE_DISABLE_DEPSET_INPUTS(StarlarkSemantics::incompatibleDisableDepsetItems),
    INCOMPATIBLE_NO_OUTPUT_ATTR_DEFAULT(StarlarkSemantics::incompatibleNoOutputAttrDefault),
    INCOMPATIBLE_NO_RULE_OUTPUTS_PARAM(StarlarkSemantics::incompatibleNoRuleOutputsParam),
    INCOMPATIBLE_NO_TARGET_OUTPUT_GROUP(StarlarkSemantics::incompatibleNoTargetOutputGroup),
    INCOMPATIBLE_NO_ATTR_LICENSE(StarlarkSemantics::incompatibleNoAttrLicense),
    INCOMPATIBLE_ALLOW_TAGS_PROPAGATION(StarlarkSemantics::experimentalAllowTagsPropagation),
    INCOMPATIBLE_REMOVE_ENABLE_TOOLCHAIN_TYPES(
        StarlarkSemantics::incompatibleRemoveEnabledToolchainTypes),
    NONE(null);

    // Using a Function here makes the enum definitions far cleaner, and, since this is
    // a private field, and we can ensure no callers treat this field as mutable.
    @SuppressWarnings("ImmutableEnumChecker")
    private final Function<StarlarkSemantics, Boolean> semanticsFunction;

    FlagIdentifier(Function<StarlarkSemantics, Boolean> semanticsFunction) {
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
   *   <li>If both parameters are {@code NONE}, this indicates the feature is not controlled by
   *       flags, and should thus be enabled.
   *   <li>If the {@code enablingFlag} parameter is non-{@code NONE}, this returns true if and only
   *       if that flag is true. (This represents a feature that is only on if a given flag is
   *       *on*).
   *   <li>If the {@code disablingFlag} parameter is non-{@code NONE}, this returns true if and only
   *       if that flag is false. (This represents a feature that is only on if a given flag is
   *       *off*).
   *   <li>It is illegal to pass both parameters as non-{@code NONE}.
   * </ul>
   */
  public boolean isFeatureEnabledBasedOnTogglingFlags(
      FlagIdentifier enablingFlag, FlagIdentifier disablingFlag) {
    Preconditions.checkArgument(
        enablingFlag == FlagIdentifier.NONE || disablingFlag == FlagIdentifier.NONE,
        "at least one of 'enablingFlag' or 'disablingFlag' must be NONE");
    if (enablingFlag != FlagIdentifier.NONE) {
      return enablingFlag.semanticsFunction.apply(this);
    } else {
      return disablingFlag == FlagIdentifier.NONE || !disablingFlag.semanticsFunction.apply(this);
    }
  }

  /** Returns the value of the given flag. */
  public boolean flagValue(FlagIdentifier flagIdentifier) {
    return flagIdentifier.semanticsFunction.apply(this);
  }

  /**
   * The AutoValue-generated concrete class implementing this one.
   *
   * <p>AutoValue implementation classes are usually package-private. We expose it here for the
   * benefit of code that relies on reflection.
   */
  public static final Class<? extends StarlarkSemantics> IMPL_CLASS =
      AutoValue_StarlarkSemantics.class;

  // <== Add new options here in alphabetic order ==>
  public abstract boolean debugDepsetDepth();

  public abstract boolean experimentalActionArgs();

  public abstract boolean experimentalAllowIncrementalRepositoryUpdates();

  public abstract boolean experimentalAspectOutputPropagation();

  public abstract boolean experimentalBuildSettingApi();

  public abstract ImmutableList<String> experimentalCcSkylarkApiEnabledPackages();

  public abstract boolean experimentalEnableAndroidMigrationApis();

  public abstract boolean experimentalGoogleLegacyApi();

  public abstract boolean experimentalNinjaActions();

  public abstract boolean experimentalPlatformsApi();

  public abstract boolean experimentalStarlarkConfigTransitions();

  public abstract boolean experimentalStarlarkUnusedInputsList();

  public abstract boolean experimentalCcSharedLibrary();

  public abstract boolean experimentalRepoRemoteExec();

  public abstract boolean incompatibleAlwaysCheckDepsetElements();

  public abstract boolean incompatibleBzlDisallowLoadAfterStatement();

  public abstract boolean incompatibleDepsetUnion();

  public abstract boolean incompatibleDisableTargetProviderFields();

  public abstract boolean incompatibleDisableThirdPartyLicenseChecking();

  public abstract boolean incompatibleDisableDeprecatedAttrParams();

  public abstract boolean incompatibleDisableDepsetItems();

  public abstract boolean incompatibleDisallowEmptyGlob();

  public abstract boolean incompatibleDisallowStructProviderSyntax();

  public abstract boolean incompatibleDisallowUnverifiedHttpDownloads();

  public abstract boolean incompatibleNewActionsApi();

  public abstract boolean incompatibleNoAttrLicense();

  public abstract boolean incompatibleNoImplicitFileExport();

  public abstract boolean incompatibleNoOutputAttrDefault();

  public abstract boolean incompatibleNoRuleOutputsParam();

  public abstract boolean incompatibleNoSupportToolsInActionInputs();

  public abstract boolean incompatibleNoTargetOutputGroup();

  public abstract boolean incompatibleRemapMainRepo();

  public abstract boolean incompatibleRemoveEnabledToolchainTypes();

  public abstract boolean incompatibleRestrictNamedParams();

  public abstract boolean incompatibleRunShellCommandString();

  public abstract boolean incompatibleVisibilityPrivateAttributesAtDefinition();

  public abstract boolean internalSkylarkFlagTestCanary();

  public abstract boolean incompatibleDoNotSplitLinkingCmdline();

  public abstract boolean incompatibleDepsetForLibrariesToLinkGetter();

  public abstract boolean incompatibleRestrictStringEscapes();

  public abstract boolean experimentalAllowTagsPropagation();

  public abstract boolean incompatibleUseCcConfigureFromRulesCc();

  @Memoized
  @Override
  public abstract int hashCode();

  /** Returns a {@link Builder} initialized with the values of this instance. */
  public abstract Builder toBuilder();

  /**
   * Returns a deterministic {@link String} representation of this object's values.
   *
   * <p>Strictly speaking, {@link AutoValue}'s generated toString implementations are unspecified.
   * Therefore it is free to e.g. randomly shuffle the order of "property=value" entries on each
   * call. In practice, it doesn't. The entries are printed in method declaration order.
   *
   * <p>We could attempt our own implementation via reflection but it's likely to be more fragile
   * than relying on the unspecified behavior to be, at least, non-pathological. YAGNI.
   */
  public String toDeterministicString() {
    return toString();
  }

  public static Builder builder() {
    return new AutoValue_StarlarkSemantics.Builder();
  }

  /** Returns a {@link Builder} initialized with default values for all options. */
  public static Builder builderWithDefaults() {
    return DEFAULT_SEMANTICS.toBuilder();
  }

  public static final StarlarkSemantics DEFAULT_SEMANTICS =
      builder()
          // <== Add new options here in alphabetic order ==>
          .debugDepsetDepth(false)
          .experimentalActionArgs(false)
          .experimentalAllowTagsPropagation(false)
          .experimentalAspectOutputPropagation(false)
          .experimentalBuildSettingApi(true)
          .experimentalCcSkylarkApiEnabledPackages(ImmutableList.of())
          .experimentalAllowIncrementalRepositoryUpdates(true)
          .experimentalEnableAndroidMigrationApis(false)
          .experimentalGoogleLegacyApi(false)
          .experimentalNinjaActions(false)
          .experimentalPlatformsApi(false)
          .experimentalStarlarkConfigTransitions(true)
          .experimentalStarlarkUnusedInputsList(true)
          .experimentalCcSharedLibrary(false)
          .experimentalRepoRemoteExec(false)
          .incompatibleAlwaysCheckDepsetElements(false)
          .incompatibleBzlDisallowLoadAfterStatement(true)
          .incompatibleDepsetUnion(true)
          .incompatibleDisableTargetProviderFields(false)
          .incompatibleDisableThirdPartyLicenseChecking(true)
          .incompatibleDisableDeprecatedAttrParams(true)
          .incompatibleDisableDepsetItems(false)
          .incompatibleDisallowEmptyGlob(false)
          .incompatibleDisallowStructProviderSyntax(false)
          .incompatibleDisallowUnverifiedHttpDownloads(true)
          .incompatibleNewActionsApi(true)
          .incompatibleNoAttrLicense(true)
          .incompatibleNoImplicitFileExport(false)
          .incompatibleNoOutputAttrDefault(true)
          .incompatibleNoRuleOutputsParam(false)
          .incompatibleNoSupportToolsInActionInputs(true)
          .incompatibleNoTargetOutputGroup(true)
          .incompatibleRemapMainRepo(true)
          .incompatibleRemoveEnabledToolchainTypes(false)
          .incompatibleRunShellCommandString(false)
          .incompatibleRestrictNamedParams(true)
          .incompatibleVisibilityPrivateAttributesAtDefinition(false)
          .internalSkylarkFlagTestCanary(false)
          .incompatibleDoNotSplitLinkingCmdline(true)
          .incompatibleDepsetForLibrariesToLinkGetter(true)
          .incompatibleRestrictStringEscapes(false)
          .incompatibleUseCcConfigureFromRulesCc(false)
          .build();

  /** Builder for {@link StarlarkSemantics}. All fields are mandatory. */
  @AutoValue.Builder
  public abstract static class Builder {

    // <== Add new options here in alphabetic order ==>
    public abstract Builder debugDepsetDepth(boolean value);

    public abstract Builder experimentalActionArgs(boolean value);

    public abstract Builder experimentalAllowIncrementalRepositoryUpdates(boolean value);

    public abstract Builder experimentalAllowTagsPropagation(boolean value);

    public abstract Builder experimentalAspectOutputPropagation(boolean value);

    public abstract Builder experimentalBuildSettingApi(boolean value);

    public abstract Builder experimentalCcSkylarkApiEnabledPackages(List<String> value);

    public abstract Builder experimentalEnableAndroidMigrationApis(boolean value);

    public abstract Builder experimentalGoogleLegacyApi(boolean value);

    public abstract Builder experimentalNinjaActions(boolean value);

    public abstract Builder experimentalPlatformsApi(boolean value);

    public abstract Builder experimentalStarlarkConfigTransitions(boolean value);

    public abstract Builder experimentalStarlarkUnusedInputsList(boolean value);

    public abstract Builder experimentalCcSharedLibrary(boolean value);

    public abstract Builder experimentalRepoRemoteExec(boolean value);

    public abstract Builder incompatibleAlwaysCheckDepsetElements(boolean value);

    public abstract Builder incompatibleBzlDisallowLoadAfterStatement(boolean value);

    public abstract Builder incompatibleDepsetUnion(boolean value);

    public abstract Builder incompatibleDisableTargetProviderFields(boolean value);

    public abstract Builder incompatibleDisableThirdPartyLicenseChecking(boolean value);

    public abstract Builder incompatibleDisableDeprecatedAttrParams(boolean value);

    public abstract Builder incompatibleDisableDepsetItems(boolean value);

    public abstract Builder incompatibleDisallowEmptyGlob(boolean value);

    public abstract Builder incompatibleDisallowStructProviderSyntax(boolean value);

    public abstract Builder incompatibleDisallowUnverifiedHttpDownloads(boolean value);

    public abstract Builder incompatibleNewActionsApi(boolean value);

    public abstract Builder incompatibleNoAttrLicense(boolean value);

    public abstract Builder incompatibleNoImplicitFileExport(boolean value);

    public abstract Builder incompatibleNoOutputAttrDefault(boolean value);

    public abstract Builder incompatibleNoRuleOutputsParam(boolean value);

    public abstract Builder incompatibleNoSupportToolsInActionInputs(boolean value);

    public abstract Builder incompatibleNoTargetOutputGroup(boolean value);

    public abstract Builder incompatibleRemapMainRepo(boolean value);

    public abstract Builder incompatibleRemoveEnabledToolchainTypes(boolean value);

    public abstract Builder incompatibleRestrictNamedParams(boolean value);

    public abstract Builder incompatibleRunShellCommandString(boolean value);

    public abstract Builder incompatibleVisibilityPrivateAttributesAtDefinition(boolean value);

    public abstract Builder internalSkylarkFlagTestCanary(boolean value);

    public abstract Builder incompatibleDoNotSplitLinkingCmdline(boolean value);

    public abstract Builder incompatibleDepsetForLibrariesToLinkGetter(boolean value);

    public abstract Builder incompatibleRestrictStringEscapes(boolean value);

    public abstract Builder incompatibleUseCcConfigureFromRulesCc(boolean value);

    public abstract StarlarkSemantics build();
  }
}
