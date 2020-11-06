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

package com.google.devtools.build.lib.analysis.platform;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Objects;
import net.starlark.java.eval.Printer;
import net.starlark.java.syntax.Location;

/** Provider for a platform constraint value that fulfills a {@link ConstraintSettingInfo}. */
@Immutable
@AutoCodec
public class ConstraintValueInfo extends NativeInfo implements ConstraintValueInfoApi {
  /** Name used in Starlark for accessing this provider. */
  public static final String STARLARK_NAME = "ConstraintValueInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<ConstraintValueInfo> PROVIDER =
      new BuiltinProvider<ConstraintValueInfo>(STARLARK_NAME, ConstraintValueInfo.class) {};

  private final ConstraintSettingInfo constraint;
  private final Label label;

  @VisibleForSerialization
  ConstraintValueInfo(ConstraintSettingInfo constraint, Label label, Location location) {
    super(location);
    this.constraint = constraint;
    this.label = label;
  }

  @Override
  public BuiltinProvider<ConstraintValueInfo> getProvider() {
    return PROVIDER;
  }

  @Override
  public ConstraintSettingInfo constraint() {
    return constraint;
  }

  @Override
  public Label label() {
    return label;
  }

  /**
   * Returns a {@link ConfigMatchingProvider} that matches if the owning target's platform includes
   * this constraint.
   *
   * <p>The {@link com.google.devtools.build.lib.rules.platform.ConstraintValue} rule can't directly
   * return a {@link ConfigMatchingProvider} because, as part of a platform's definition, it doesn't
   * have access to the platform during its analysis.
   *
   * <p>Instead, a target with a <code>select()</code> on a {@link
   * com.google.devtools.build.lib.rules.platform.ConstraintValue} passes its platform info to this
   * method.
   */
  public ConfigMatchingProvider configMatchingProvider(PlatformInfo platformInfo) {
    return new ConfigMatchingProvider(
        label,
        ImmutableMultimap.of(),
        ImmutableMap.of(),
        // Technically a select() on a constraint_value requires PlatformConfiguration, since that's
        // used to build the platform this checks against. But platformInfo's existence implies
        // the owning target already depends on PlatformConfiguration. And we can't reference
        // PlatformConfiguration.class here without a build dependency cycle.
        ImmutableSet.of(),
        platformInfo.constraints().hasConstraintValue(this));
  }

  @Override
  public void repr(Printer printer) {
    Printer.format(
        printer,
        "ConstraintValueInfo(setting=%s, %s)",
        constraint.label().toString(),
        label.toString());
  }

  /** Returns a new {@link ConstraintValueInfo} with the given data. */
  public static ConstraintValueInfo create(ConstraintSettingInfo constraint, Label value) {
    return create(constraint, value, Location.BUILTIN);
  }

  /** Returns a new {@link ConstraintValueInfo} with the given data. */
  public static ConstraintValueInfo create(
      ConstraintSettingInfo constraint, Label value, Location location) {
    return new ConstraintValueInfo(constraint, value, location);
  }

  /** Add this constraint value to the given fingerprint. */
  public void addTo(Fingerprint fp) {
    this.constraint.addTo(fp);
    fp.addString(label.getCanonicalForm());
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof ConstraintValueInfo)) {
      return false;
    }
    ConstraintValueInfo that = (ConstraintValueInfo) o;
    return Objects.equals(constraint, that.constraint)
        && Objects.equals(label, that.label);
  }

  @Override
  public int hashCode() {
    return Objects.hash(constraint, label);
  }
}
