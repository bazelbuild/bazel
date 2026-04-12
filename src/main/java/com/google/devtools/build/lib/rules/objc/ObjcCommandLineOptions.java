// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;

/** Command-line options for building Objective-C targets. */
public class ObjcCommandLineOptions extends FragmentOptions {
  @Option(
    name = "device_debug_entitlements",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.SIGNING,
    effectTags = {OptionEffectTag.CHANGES_INPUTS},
    help =
        "If set, and compilation mode is not 'opt', objc apps will include debug entitlements "
            + "when signing."
  )
  public boolean deviceDebugEntitlements;

  @Option(
      name = "incompatible_disallow_sdk_frameworks_attributes",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If true, disallow sdk_frameworks and weak_sdk_frameworks attributes in objc_library and"
              + " objc_import.")
  public boolean incompatibleDisallowSdkFrameworksAttributes;

  @Option(
      name = "incompatible_objc_alwayslink_by_default",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If true, make the default value true for alwayslink attributes in objc_library and"
              + " objc_import.")
  public boolean incompatibleObjcAlwayslinkByDefault;

  /**
   * @deprecated delete when we are sure it's not used anywhere.
   */
  @Deprecated
  @Option(
      name = "incompatible_disable_native_apple_binary_rule",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {
        OptionEffectTag.EAGERNESS_TO_EXIT,
      },
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE, OptionMetadataTag.DEPRECATED},
      help = "No-op. Kept here for backwards compatibility.")
  public boolean incompatibleDisableNativeAppleBinaryRule;

  @Option(
      name = "incompatible_strip_executable_safely",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If true, strip action for executables will use flag -x, which does not break dynamic "
              + "symbol resolution.")
  public boolean incompatibleStripExecutableSafely;

  @Option(
      name = "incompatible_builtin_objc_strip_action",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
      effectTags = {OptionEffectTag.ACTION_COMMAND_LINES},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help = "Whether to emit a strip action as part of objc linking.")
  public boolean incompatibleBuiltinObjcStripAction;

  // Tracked in #28082.
  @Option(
      name = "incompatible_remove_ctx_objc_fragment",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Apple build flags are defined with Apple rules (in BUIILD files) and"
              + " ctx.fragments.objc is undefined. This is a migration flag to move all Apple"
              + " flags from core Bazel to Apple rules.")
  public boolean disableObjcFragment;
}
