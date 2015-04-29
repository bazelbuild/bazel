// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.List;

/**
 * Provider that describes a simulator device.
 */
@Immutable
public final class IosDeviceProvider implements TransitiveInfoProvider {
  /** A builder of {@link IosDeviceProvider}s. */
  public static final class Builder {
    private String type;
    private String iosVersion;
    private String locale;

    public Builder setType(String type) {
      this.type = type;
      return this;
    }

    public Builder setIosVersion(String iosVersion) {
      this.iosVersion = iosVersion;
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
  private final String iosVersion;
  private final String locale;

  private IosDeviceProvider(Builder builder) {
    this.type = Preconditions.checkNotNull(builder.type);
    this.iosVersion = Preconditions.checkNotNull(builder.iosVersion);
    this.locale = Preconditions.checkNotNull(builder.locale);
  }

  public String getType() {
    return type;
  }

  public String getIosVersion() {
    return iosVersion;
  }

  public String getLocale() {
    return locale;
  }

  /**
   * Returns a list of substitutions which should be performed to the test runner script, to fill
   * in device-specific data which may be required in order to run tests.
   */
  public List<Substitution> getSubstitutionsForTestRunnerScript() {
    return ImmutableList.of(
        Substitution.of("%(device_type)s", getType()),
        Substitution.of("%(simulator_sdk)s", getIosVersion())
    );
  }
}
