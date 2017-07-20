// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Preconditions;

/** Supplies information needed when a dependency serves as an {@code xctest_app}. */
@Immutable
@SkylarkModule(
  name = "XcTestAppProvider",
  category = SkylarkModuleCategory.PROVIDER,
  doc =
      "Deprecated. A provider for XCTest apps for testing. This is a legacy provider and should "
          + "not be used."
)
public final class XcTestAppProvider extends SkylarkClassObject
    implements TransitiveInfoProvider, TransitiveInfoProvider.WithLegacySkylarkName {
  /**
   * The skylark struct key name for a rule implementation to use when exporting an ObjcProvider.
   */
  public static final String XCTEST_APP_SKYLARK_PROVIDER_NAME = "xctest_app";

  @Override
  public String getSkylarkName() {
    return XCTEST_APP_SKYLARK_PROVIDER_NAME;
  }

  private static final NativeClassObjectConstructor<XcTestAppProvider> XCTEST_APP_PROVIDER =
      new NativeClassObjectConstructor<XcTestAppProvider>(
          XcTestAppProvider.class, "xctest_app_provider") {
        @Override
        public String getErrorMessageFormatForInstances() {
          return "XcTestAppProvider field %s could not be instantiated";
        }
      };

  private final Artifact bundleLoader;
  private final Artifact ipa;
  private final ObjcProvider objcProvider;

  XcTestAppProvider(Artifact bundleLoader, Artifact ipa, ObjcProvider objcProvider) {
    super(XCTEST_APP_PROVIDER, getSkylarkFields(bundleLoader, ipa, objcProvider));
    this.bundleLoader = Preconditions.checkNotNull(bundleLoader);
    this.ipa = Preconditions.checkNotNull(ipa);
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
  }

  /** The bundle loader, which corresponds to the test app's binary. */
  public Artifact getBundleLoader() {
    return bundleLoader;
  }

  /** The test app's IPA. */
  public Artifact getIpa() {
    return ipa;
  }

  /**
   * An {@link ObjcProvider} that should be included by any test target that uses this app as its
   * {@code xctest_app}. This is <strong>not</strong> a typical {@link ObjcProvider} - it has
   * certain linker-releated keys omitted, such as {@link ObjcProvider#LIBRARY}, since XcTests have
   * access to symbols in their test rig without linking them into the main test binary.
   *
   * <p>The current list of whitelisted values can be found in {@link
   * ReleaseBundlingSupport#xcTestAppProvider}.
   */
  public ObjcProvider getObjcProvider() {
    return objcProvider;
  }

  private static ImmutableMap<String, Object> getSkylarkFields(
      Artifact bundleLoader, Artifact ipa, ObjcProvider objcProvider) {
    return new ImmutableMap.Builder<String, Object>()
        .put("bundle_loader", bundleLoader)
        .put("ipa", ipa)
        .put("objc", objcProvider)
        .build();
  }
}
