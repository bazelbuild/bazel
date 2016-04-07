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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.util.Preconditions;

import javax.annotation.Nullable;

/**
 * Provider that describes a simulator device.
 */
@Immutable
public final class IosDeviceProvider implements TransitiveInfoProvider {
  /** A builder of {@link IosDeviceProvider}s. */
  public static final class Builder {
    private String type;
    private DottedVersion iosVersion;
    private String locale;
    @Nullable
    private DottedVersion xcodeVersion;

    /**
     * Sets the hardware type of the device, corresponding to the {@code simctl} device type.
     */
    public Builder setType(String type) {
      this.type = type;
      return this;
    }

    /**
     * Sets the iOS version of the simulator to use. This may be different than the iOS sdk version
     * used to build the application.
     */
    public Builder setIosVersion(DottedVersion iosVersion) {
      this.iosVersion = iosVersion;
      return this;
    }

    /**
     * Sets the xcode version to obtain the iOS simulator from. This may be different than the
     * xcode version with which the application was built.
     */
    public Builder setXcodeVersion(@Nullable DottedVersion xcodeVersion) {
      this.xcodeVersion = xcodeVersion;
      return this;
    }

    public Builder setLocale(String locale) {
      this.locale = locale;
      return this;
    }

    public IosDeviceProvider build() {
      return new IosDeviceProvider(this);
    }
  }

  private final String type;
  private final DottedVersion iosVersion;
  private final DottedVersion xcodeVersion;
  private final String locale;

  private IosDeviceProvider(Builder builder) {
    this.type = Preconditions.checkNotNull(builder.type);
    this.iosVersion = Preconditions.checkNotNull(builder.iosVersion);
    this.locale = Preconditions.checkNotNull(builder.locale);
    this.xcodeVersion = builder.xcodeVersion;
  }

  public String getType() {
    return type;
  }

  public DottedVersion getIosVersion() {
    return iosVersion;
  }

  @Nullable
  public DottedVersion getXcodeVersion() {
    return xcodeVersion;
  }

  public String getLocale() {
    return locale;
  }

  /**
   * Returns an {@code IosTestSubstitutionProvider} exposing substitutions indicating how to run a
   * test in this particular iOS simulator configuration.
   */
  public IosTestSubstitutionProvider iosTestSubstitutionProvider() {
    return new IosTestSubstitutionProvider(
        ImmutableList.of(
            Substitution.of("%(device_type)s", getType()),
            Substitution.of("%(simulator_sdk)s", getIosVersion().toString()),
            Substitution.of("%(locale)s", getLocale())));
  }
}
