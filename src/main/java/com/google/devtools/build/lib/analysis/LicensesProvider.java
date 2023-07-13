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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.License;
import java.util.Objects;

/** A {@link ConfiguredTarget} that has licensed targets in its transitive closure. */
public interface LicensesProvider extends Info {
  /**
   * The set of label - license associations in the transitive closure.
   *
   * <p>Always returns an empty set if {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfigurationValue#checkLicenses()} is
   * false.
   */
  NestedSet<TargetLicense> getTransitiveLicenses();

  /**
   * A label - license association for output_licenses. If there are no output_licenses it returns
   * null.
   */
  TargetLicense getOutputLicenses();

  /**
   * Return whether there is an output_licenses.
   */
  boolean hasOutputLicenses();

  public static final BuiltinProvider<LicensesProvider> PROVIDER =
      new BuiltinProvider<LicensesProvider>("LicenseInfo", LicensesProvider.class) {};

  /** License association for a particular target. */
  final class TargetLicense {
    private final Label label;
    private final License license;

    public TargetLicense(Label label, License license) {
      Preconditions.checkNotNull(label);
      Preconditions.checkNotNull(license);
      this.label = label;
      this.license = license;
    }

    /**
     * Returns the label of the associated target.
     */
    public Label getLabel() {
      return label;
    }

    /**
     * Returns the license for the target.
     */
    public License getLicense() {
      return license;
    }

    @Override
    public int hashCode() {
      return Objects.hash(label, license);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof TargetLicense)) {
        return false;
      }
      TargetLicense other = (TargetLicense) obj;
      return label.equals(other.label) && license.equals(other.license);
    }

    @Override
    public String toString() {
      return label + " => " + license;
    }
  }
}
