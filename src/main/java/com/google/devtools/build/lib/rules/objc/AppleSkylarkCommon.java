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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.rules.apple.XcodeVersionProperties;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import java.util.Map.Entry;
import javax.annotation.Nullable;

/**
 * A class that exposes apple rule implementation internals to skylark.
 */
@SkylarkModule(
  name = "apple_common",
  doc = "Functions for skylark to access internals of the apple rule implementations."
)
public class AppleSkylarkCommon {
 
  @VisibleForTesting
  public static final String BAD_KEY_ERROR = "Argument %s not a recognized key, 'providers',"
      + " or 'direct_dep_providers'.";

  @VisibleForTesting
  public static final String BAD_SET_TYPE_ERROR =
      "Value for key %s must be a set of %s, instead found set of %s.";

  @VisibleForTesting
  public static final String BAD_PROVIDERS_ITER_ERROR =
      "Value for argument 'providers' must be a list of ObjcProvider instances, instead found %s.";

  @VisibleForTesting
  public static final String BAD_PROVIDERS_ELEM_ERROR =
      "Value for argument 'providers' must be a list of ObjcProvider instances, instead found "
          + "iterable with %s.";

  @VisibleForTesting
  public static final String NOT_SET_ERROR = "Value for key %s must be a set, instead found %s.";

  @VisibleForTesting
  public static final String MISSING_KEY_ERROR = "No value for required key %s was present.";

  @Nullable
  private SkylarkClassObject platformType;
  @Nullable
  private SkylarkClassObject platform;

  @SkylarkCallable(
      name = "apple_toolchain",
      doc = "Utilities for resolving items from the apple toolchain."
  )
  public AppleToolchain getAppleToolchain() {
    return new AppleToolchain();
  }

  @SkylarkCallable(
    name = "platform_type",
    doc = "Returns a struct containing fields corresponding to Apple platform types: 'ios', "
        + "'watchos', 'tvos', and 'macos'. These values can be passed to methods that expect a "
        + "platform type, like the 'apple' configuration fragment's 'multi_arch_platform' "
        + "method. For example, ctx.fragments.apple.multi_arch_platform(apple_common."
        + "platform_type.ios).",
    structField = true
  )
  public SkylarkClassObject getPlatformTypeStruct() {
    if (platformType == null) {
      platformType = PlatformType.getSkylarkStruct();
    }
    return platformType;
  }

  @SkylarkCallable(
      name = "platform",
      doc = "Returns a struct containing fields corresponding to Apple platforms. These values "
          + "can be passed to methods that expect a platform, like the 'apple' configuration "
          + "fragment's 'sdk_version_for_platform' method. Each platform_type except for macosx "
          + "has two platform types -- one for device, and one for simulator.",
      structField = true
  )
  public SkylarkClassObject getPlatformStruct() {
    if (platform == null) {
      platform = Platform.getSkylarkStruct();
    }
    return platform;
  }

  @SkylarkCallable(
    name = XcodeVersionProperties.SKYLARK_NAME,
    doc =
        "Returns the provider constructor for XcodeVersionProperties. If a target propagates "
            + "the XcodeVersionProperties provider, use this as the key with which to retrieve it.",
    structField = true
  )
  public ClassObjectConstructor getXcodeVersionPropertiesConstructor() {
    return XcodeVersionProperties.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = AppleDylibBinaryProvider.SKYLARK_NAME,
    doc =
        "Returns the provider constructor for AppleDylibBinary. If a target propagates "
            + "the AppleDylibBinary provider, use this as the key with which to retrieve it.",
    structField = true
  )
  public ClassObjectConstructor getAppleDylibBinaryConstructor() {
    return AppleDylibBinaryProvider.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = AppleExecutableBinaryProvider.SKYLARK_NAME,
    doc =
        "Returns the provider constructor for AppleExecutableBinary. If a target propagates "
            + "the AppleExecutableBinary provider, use this as the key with which to retrieve it.",
    structField = true
  )
  public ClassObjectConstructor getAppleExecutableBinaryConstructor() {
    return AppleExecutableBinaryProvider.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = AppleLoadableBundleBinaryProvider.SKYLARK_NAME,
    doc =
        "Returns the provider constructor for AppleLoadableBundleBinaryProvider. If a target "
            + "propagates the AppleLoadableBundleBinaryProvider provider, use this as the key "
            + "with which to retrieve it.",
    structField = true
  )
  public ClassObjectConstructor getAppleLoadableBundleBinaryConstructor() {
    return AppleLoadableBundleBinaryProvider.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = IosDeviceProvider.SKYLARK_NAME,
    doc =
        "Returns the provider constructor for IosDeviceProvider. Use this as a key to access the "
            + "attributes exposed by ios_device.",
    structField = true
  )
  public ClassObjectConstructor getIosDeviceProviderConstructor() {
    return IosDeviceProvider.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkSignature(
    name = "new_objc_provider",
    objectType = AppleSkylarkCommon.class,
    returnType = ObjcProvider.class,
    doc = "Creates a new ObjcProvider instance.",
    parameters = {
      @Param(name = "self", type = AppleSkylarkCommon.class, doc = "The apple_common instance."),
      @Param(
        name = "uses_swift",
        type = Boolean.class,
        defaultValue = "False",
        named = true,
        positional = false,
        doc = "Whether this provider should enable Swift support."
      )
    },
    extraKeywords =
        @Param(
          name = "kwargs",
          type = SkylarkDict.class,
          defaultValue = "{}",
          doc = "Dictionary of arguments."
        )
  )
  public static final BuiltinFunction NEW_OBJC_PROVIDER =
      new BuiltinFunction("new_objc_provider") {
        @SuppressWarnings("unused")
        // This method is registered statically for skylark, and never called directly.
        public ObjcProvider invoke(
            AppleSkylarkCommon self, Boolean usesSwift, SkylarkDict<String, Object> kwargs) {
          ObjcProvider.Builder resultBuilder = new ObjcProvider.Builder();
          if (usesSwift) {
            resultBuilder.add(ObjcProvider.FLAG, ObjcProvider.Flag.USES_SWIFT);
          }
          for (Entry<String, Object> entry : kwargs.entrySet()) {
            Key<?> key = ObjcProvider.getSkylarkKeyForString(entry.getKey());
            if (key != null) {
              resultBuilder.addElementsFromSkylark(key, entry.getValue());
            } else if (entry.getKey().equals("providers")) {
              resultBuilder.addProvidersFromSkylark(entry.getValue());
            } else if (entry.getKey().equals("direct_dep_providers")) {
              resultBuilder.addDirectDepProvidersFromSkylark(entry.getValue());
            } else {
              throw new IllegalArgumentException(String.format(BAD_KEY_ERROR, entry.getKey()));
            }
          }
          return resultBuilder.build();
        }
      };

  @SkylarkSignature(
    name = "new_xctest_app_provider",
    objectType = AppleSkylarkCommon.class,
    returnType = XcTestAppProvider.class,
    doc = "Creates a new XcTestAppProvider instance.",
    parameters = {
      @Param(name = "self", type = AppleSkylarkCommon.class, doc = "The apple_common instance."),
      @Param(
        name = "bundle_loader",
        type = Artifact.class,
        named = true,
        positional = false,
        doc = "The bundle loader for the test. Corresponds to the binary inside the test IPA."
      ),
      @Param(
        name = "ipa",
        type = Artifact.class,
        named = true,
        positional = false,
        doc = "The test IPA."
      ),
      @Param(
        name = "objc_provider",
        type = ObjcProvider.class,
        named = true,
        positional = false,
        doc = "An ObjcProvider that should be included by tests using this test bundle."
      )
    }
  )
  public static final BuiltinFunction NEW_XCTEST_APP_PROVIDER =
      new BuiltinFunction("new_xctest_app_provider") {
        @SuppressWarnings("unused")
        // This method is registered statically for skylark, and never called directly.
        public XcTestAppProvider invoke(
            AppleSkylarkCommon self,
            Artifact bundleLoader,
            Artifact ipa,
            ObjcProvider objcProvider) {
          return new XcTestAppProvider(bundleLoader, ipa, objcProvider);
        }
      };

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(AppleSkylarkCommon.class);
  }
}

