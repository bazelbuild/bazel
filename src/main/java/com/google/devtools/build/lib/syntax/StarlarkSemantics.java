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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.List;

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
   * A set of names of boolean application flags each corresponding to a StarlarkSemantics feature.
   */
  // TODO(adonovan): StarlarkSemantics, being part of the core frontend, shouldn't refer to Bazel
  // features. There's no need for an enumeration to represent a set of boolean features. Instead,
  // have StarlarkSemantics hold a set of enabled features (strings), and have callers query
  // features by name. The features can be named string constants, defined close to the code they
  // affect, to avoid accidential misspellings.
  public static final class FlagIdentifier {
    private FlagIdentifier() {} // uninstantiable

    // The strings here match the names of the StarlarkSemantics methods,
    // which in turn match the actual flag names; they should be kept
    // consistent as they may appear in error messages.
    // TODO(adonovan): move these constants up into the relevant packages of
    // Bazel, and make them identical to the strings used in flag declarations.
    public static final String EXPERIMENTAL_ACTION_ARGS = "experimental_action_args";
    public static final String EXPERIMENTAL_ALLOW_INCREMENTAL_REPOSITORY_UPDATES =
        "experimental_allow_incremental_repository_updates";
    public static final String EXPERIMENTAL_DISABLE_EXTERNAL_PACKGE =
        "experimental_disable_external_package";
    public static final String EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT =
        "experimental_sibling_repository_layout";
    public static final String EXPERIMENTAL_ASPECT_OUTPUT_PROPAGATION =
        "experimental_aspect_output_propagation";
    public static final String EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS =
        "experimental_enable_android_migration_apis";
    public static final String EXPERIMENTAL_BUILD_SETTING_API = "experimental_build_setting_api";
    public static final String EXPERIMENTAL_GOOGLE_LEGACY_API = "experimental_google_legacy_api";
    public static final String EXPERIMENTAL_NINJA_ACTIONS = "experimental_ninja_actions";
    public static final String EXPERIMENTAL_PLATFORM_API = "experimental_platform_api";
    public static final String EXPERIMENTAL_STARLARK_CONFIG_TRANSITION =
        "experimental_starlark_config_transition";
    public static final String EXPERIMENTAL_STARLARK_UNUSED_INPUTS_LIST =
        "experimental_starlark_unused_inputs_list";
    public static final String EXPERIMENTAL_REPO_REMOTE_EXEC = "experimental_repo_remote_exec";
    public static final String INCOMPATIBLE_APPLICABLE_LICENSES =
        "incompatible_applicable_licenses";
    public static final String INCOMPATIBLE_DISABLE_DEPSET_INPUTS =
        "incompatible_disable_depset_inputs";
    public static final String INCOMPATIBLE_NO_OUTPUT_ATTR_DEFAULT =
        "incompatible_no_output_attr_default";
    public static final String INCOMPATIBLE_NO_RULE_OUTPUTS_PARAM =
        "incompatible_no_rule_outputs_param";
    public static final String INCOMPATIBLE_NO_TARGET_OUTPUT_GROUP =
        "incompatible_no_target_output_group";
    public static final String INCOMPATIBLE_NO_ATTR_LICENSE = "incompatible_no_attr_license";
    public static final String INCOMPATIBLE_ALLOW_TAGS_PROPAGATION =
        "incompatible_allow_tags_propagation";
    public static final String INCOMPATIBLE_REMOVE_ENABLE_TOOLCHAIN_TYPES =
        "incompatible_remove_enable_toolchain_types";
    public static final String INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API =
        "incompatible_require_linker_input_cc_api";
  }

  // TODO(adonovan): replace the fields of StarlarkSemantics
  // by a map from string to object, and make it the clients's job
  // to know the type. This function would then become simply:
  //  return Boolean.TRUE.equals(map.get(flag)).
  boolean flagValue(String flag) {
    switch (flag) {
      case FlagIdentifier.EXPERIMENTAL_ACTION_ARGS:
        return experimentalActionArgs();
      case FlagIdentifier.EXPERIMENTAL_ALLOW_INCREMENTAL_REPOSITORY_UPDATES:
        return experimentalAllowIncrementalRepositoryUpdates();
      case FlagIdentifier.EXPERIMENTAL_DISABLE_EXTERNAL_PACKGE:
        return experimentalDisableExternalPackage();
      case FlagIdentifier.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT:
        return experimentalSiblingRepositoryLayout();
      case FlagIdentifier.EXPERIMENTAL_ASPECT_OUTPUT_PROPAGATION:
        return experimentalAspectOutputPropagation();
      case FlagIdentifier.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS:
        return experimentalEnableAndroidMigrationApis();
      case FlagIdentifier.EXPERIMENTAL_BUILD_SETTING_API:
        return experimentalBuildSettingApi();
      case FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API:
        return experimentalGoogleLegacyApi();
      case FlagIdentifier.EXPERIMENTAL_NINJA_ACTIONS:
        return experimentalNinjaActions();
      case FlagIdentifier.EXPERIMENTAL_PLATFORM_API:
        return experimentalPlatformsApi();
      case FlagIdentifier.EXPERIMENTAL_STARLARK_CONFIG_TRANSITION:
        return experimentalStarlarkConfigTransitions();
      case FlagIdentifier.EXPERIMENTAL_STARLARK_UNUSED_INPUTS_LIST:
        return experimentalStarlarkUnusedInputsList();
      case FlagIdentifier.EXPERIMENTAL_REPO_REMOTE_EXEC:
        return experimentalRepoRemoteExec();
      case FlagIdentifier.INCOMPATIBLE_APPLICABLE_LICENSES:
        return incompatibleApplicableLicenses();
      case FlagIdentifier.INCOMPATIBLE_DISABLE_DEPSET_INPUTS:
        return incompatibleDisableDepsetItems();
      case FlagIdentifier.INCOMPATIBLE_NO_OUTPUT_ATTR_DEFAULT:
        return incompatibleNoOutputAttrDefault();
      case FlagIdentifier.INCOMPATIBLE_NO_RULE_OUTPUTS_PARAM:
        return incompatibleNoRuleOutputsParam();
      case FlagIdentifier.INCOMPATIBLE_NO_TARGET_OUTPUT_GROUP:
        return incompatibleNoTargetOutputGroup();
      case FlagIdentifier.INCOMPATIBLE_NO_ATTR_LICENSE:
        return incompatibleNoAttrLicense();
      case FlagIdentifier.INCOMPATIBLE_ALLOW_TAGS_PROPAGATION:
        return experimentalAllowTagsPropagation();
      case FlagIdentifier.INCOMPATIBLE_REMOVE_ENABLE_TOOLCHAIN_TYPES:
        return incompatibleRemoveEnabledToolchainTypes();
      case FlagIdentifier.INCOMPATIBLE_REQUIRE_LINKER_INPUT_CC_API:
        return incompatibleRequireLinkerInputCcApi();
      default:
        throw new IllegalArgumentException(flag);
    }
  }

  /**
   * Returns true if a feature attached to the given toggling flags should be enabled.
   *
   * <ul>
   *   <li>If both parameters are empty, this indicates the feature is not controlled by flags, and
   *       should thus be enabled.
   *   <li>If the {@code enablingFlag} parameter is non-empty, this returns true if and only if that
   *       flag is true. (This represents a feature that is only on if a given flag is *on*).
   *   <li>If the {@code disablingFlag} parameter is non-empty, this returns true if and only if
   *       that flag is false. (This represents a feature that is only on if a given flag is *off*).
   *   <li>It is illegal to pass both parameters as non-empty.
   * </ul>
   */
  boolean isFeatureEnabledBasedOnTogglingFlags(String enablingFlag, String disablingFlag) {
    Preconditions.checkArgument(
        enablingFlag.isEmpty() || disablingFlag.isEmpty(),
        "at least one of 'enablingFlag' or 'disablingFlag' must be empty");
    if (!enablingFlag.isEmpty()) {
      return this.flagValue(enablingFlag);
    } else if (!disablingFlag.isEmpty()) {
      return !this.flagValue(disablingFlag);
    } else {
      return true;
    }
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

  public abstract boolean experimentalDisableExternalPackage();

  public abstract boolean experimentalSiblingRepositoryLayout();

  public abstract boolean incompatibleAlwaysCheckDepsetElements();

  public abstract boolean incompatibleApplicableLicenses();

  public abstract boolean incompatibleBzlDisallowLoadAfterStatement();

  public abstract boolean incompatibleDepsetUnion();

  public abstract boolean incompatibleDisableTargetProviderFields();

  public abstract boolean incompatibleDisableThirdPartyLicenseChecking();

  public abstract boolean incompatibleDisableDeprecatedAttrParams();

  public abstract boolean incompatibleDisableDepsetItems();

  public abstract boolean incompatibleDisallowEmptyGlob();

  public abstract boolean incompatibleDisallowStructProviderSyntax();

  public abstract boolean incompatibleNewActionsApi();

  public abstract boolean incompatibleNoAttrLicense();

  public abstract boolean incompatibleNoImplicitFileExport();

  public abstract boolean incompatibleNoOutputAttrDefault();

  public abstract boolean incompatibleNoRuleOutputsParam();

  public abstract boolean incompatibleNoSupportToolsInActionInputs();

  public abstract boolean incompatibleNoTargetOutputGroup();

  public abstract boolean incompatibleRemoveEnabledToolchainTypes();

  public abstract boolean incompatibleRestrictNamedParams();

  public abstract boolean incompatibleRunShellCommandString();

  public abstract boolean incompatibleVisibilityPrivateAttributesAtDefinition();

  public abstract boolean internalSkylarkFlagTestCanary();

  public abstract boolean incompatibleDoNotSplitLinkingCmdline();

  public abstract boolean incompatibleDepsetForLibrariesToLinkGetter();

  public abstract boolean incompatibleRequireLinkerInputCcApi();

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
          .experimentalDisableExternalPackage(false)
          .experimentalSiblingRepositoryLayout(false)
          .incompatibleAlwaysCheckDepsetElements(true)
          .incompatibleApplicableLicenses(false)
          .incompatibleBzlDisallowLoadAfterStatement(true)
          .incompatibleDepsetUnion(true)
          .incompatibleDisableTargetProviderFields(false)
          .incompatibleDisableThirdPartyLicenseChecking(true)
          .incompatibleDisableDeprecatedAttrParams(true)
          .incompatibleDisableDepsetItems(false)
          .incompatibleDisallowEmptyGlob(false)
          .incompatibleDisallowStructProviderSyntax(false)
          .incompatibleNewActionsApi(true)
          .incompatibleNoAttrLicense(true)
          .incompatibleNoImplicitFileExport(false)
          .incompatibleNoOutputAttrDefault(true)
          .incompatibleNoRuleOutputsParam(false)
          .incompatibleNoSupportToolsInActionInputs(true)
          .incompatibleNoTargetOutputGroup(true)
          .incompatibleRemoveEnabledToolchainTypes(true)
          .incompatibleRunShellCommandString(false)
          .incompatibleRestrictNamedParams(true)
          .incompatibleVisibilityPrivateAttributesAtDefinition(false)
          .internalSkylarkFlagTestCanary(false)
          .incompatibleDoNotSplitLinkingCmdline(true)
          .incompatibleDepsetForLibrariesToLinkGetter(true)
          .incompatibleRequireLinkerInputCcApi(false)
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

    public abstract Builder experimentalDisableExternalPackage(boolean value);

    public abstract Builder experimentalSiblingRepositoryLayout(boolean value);

    public abstract Builder incompatibleAlwaysCheckDepsetElements(boolean value);

    public abstract Builder incompatibleApplicableLicenses(boolean value);

    public abstract Builder incompatibleBzlDisallowLoadAfterStatement(boolean value);

    public abstract Builder incompatibleDepsetUnion(boolean value);

    public abstract Builder incompatibleDisableTargetProviderFields(boolean value);

    public abstract Builder incompatibleDisableThirdPartyLicenseChecking(boolean value);

    public abstract Builder incompatibleDisableDeprecatedAttrParams(boolean value);

    public abstract Builder incompatibleDisableDepsetItems(boolean value);

    public abstract Builder incompatibleDisallowEmptyGlob(boolean value);

    public abstract Builder incompatibleDisallowStructProviderSyntax(boolean value);

    public abstract Builder incompatibleNewActionsApi(boolean value);

    public abstract Builder incompatibleNoAttrLicense(boolean value);

    public abstract Builder incompatibleNoImplicitFileExport(boolean value);

    public abstract Builder incompatibleNoOutputAttrDefault(boolean value);

    public abstract Builder incompatibleNoRuleOutputsParam(boolean value);

    public abstract Builder incompatibleNoSupportToolsInActionInputs(boolean value);

    public abstract Builder incompatibleNoTargetOutputGroup(boolean value);

    public abstract Builder incompatibleRemoveEnabledToolchainTypes(boolean value);

    public abstract Builder incompatibleRestrictNamedParams(boolean value);

    public abstract Builder incompatibleRunShellCommandString(boolean value);

    public abstract Builder incompatibleVisibilityPrivateAttributesAtDefinition(boolean value);

    public abstract Builder internalSkylarkFlagTestCanary(boolean value);

    public abstract Builder incompatibleDoNotSplitLinkingCmdline(boolean value);

    public abstract Builder incompatibleDepsetForLibrariesToLinkGetter(boolean value);

    public abstract Builder incompatibleRequireLinkerInputCcApi(boolean value);

    public abstract Builder incompatibleRestrictStringEscapes(boolean value);

    public abstract Builder incompatibleUseCcConfigureFromRulesCc(boolean value);

    public abstract StarlarkSemantics build();
  }
}
