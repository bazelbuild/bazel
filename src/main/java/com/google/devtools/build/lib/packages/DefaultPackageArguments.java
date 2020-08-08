// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;
import java.util.List;
import java.util.Set;

/** Encapsulates the core, default set of {@link PackageArgument}s. */
final class DefaultPackageArguments {

  private DefaultPackageArguments() {}

  /** Returns the default set of {@link PackageArgument}s. */
  static ImmutableList<PackageArgument<?>> get() {
    return ImmutableList.of(
            new DefaultDeprecation(),
            new DefaultDistribs(),
            new DefaultApplicableLicenses(),
            new DefaultLicenses(),
            new DefaultTestOnly(),
            new DefaultVisibility(),
            new Features(),
            new DefaultCompatibleWith(),
            new DefaultRestrictedTo());
  }

  private static class DefaultVisibility extends PackageArgument<List<Label>> {
    private DefaultVisibility() {
      super("default_visibility", BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) throws EvalException {
      pkgBuilder.setDefaultVisibility(
          PackageUtils.getVisibility(pkgBuilder.getBuildFileLabel(), value));
    }
  }

  private static class DefaultTestOnly extends PackageArgument<Boolean> {
    private DefaultTestOnly() {
      super("default_testonly", Type.BOOLEAN);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        Boolean value) {
      pkgBuilder.setDefaultTestonly(value);
    }
  }

  private static class DefaultDeprecation extends PackageArgument<String> {
    private DefaultDeprecation() {
      super("default_deprecation", Type.STRING);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        String value) {
      pkgBuilder.setDefaultDeprecation(value);
    }
  }

  private static class Features extends PackageArgument<List<String>> {
    private Features() {
      super("features", Type.STRING_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<String> value) {
      pkgBuilder.addFeatures(value);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#APPLICABLE_LICENSES_ATTR} when not explicitly
   * specified.
   */
  private static class DefaultApplicableLicenses extends PackageArgument<List<Label>> {
    private static final String DEFAULT_APPLICABLE_LICENSES_ATTRIBUTE =
        "default_applicable_licenses";

    private DefaultApplicableLicenses() {
      super(DEFAULT_APPLICABLE_LICENSES_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, List<Label> value) {
      pkgBuilder.setDefaultApplicableLicenses(
          value, DEFAULT_APPLICABLE_LICENSES_ATTRIBUTE, location);
    }
  }

  private static class DefaultLicenses extends PackageArgument<License> {
    private DefaultLicenses() {
      super("licenses", BuildType.LICENSE);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        License value) {
      pkgBuilder.setDefaultLicense(value);
    }
  }

  private static class DefaultDistribs extends PackageArgument<Set<DistributionType>> {
    private DefaultDistribs() {
      super("distribs", BuildType.DISTRIBUTIONS);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        Set<DistributionType> value) {
      pkgBuilder.setDefaultDistribs(value);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#COMPATIBLE_ENVIRONMENT_ATTR} when not
   * explicitly specified.
   */
  private static class DefaultCompatibleWith extends PackageArgument<List<Label>> {
    private static final String DEFAULT_COMPATIBLE_WITH_ATTRIBUTE = "default_compatible_with";

    private DefaultCompatibleWith() {
      super(DEFAULT_COMPATIBLE_WITH_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultCompatibleWith(value, DEFAULT_COMPATIBLE_WITH_ATTRIBUTE, location);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#RESTRICTED_ENVIRONMENT_ATTR} when not
   * explicitly specified.
   */
  private static class DefaultRestrictedTo extends PackageArgument<List<Label>> {
    private static final String DEFAULT_RESTRICTED_TO_ATTRIBUTE = "default_restricted_to";

    private DefaultRestrictedTo() {
      super(DEFAULT_RESTRICTED_TO_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location,
        List<Label> value) {
      pkgBuilder.setDefaultRestrictedTo(value, DEFAULT_RESTRICTED_TO_ATTRIBUTE, location);
    }
  }
}
