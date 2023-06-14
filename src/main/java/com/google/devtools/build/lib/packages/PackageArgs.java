// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.config.FeatureSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * A group of {@link Package} argument values that may be provided by `package()` or `repo()` calls.
 */
@AutoValue
public abstract class PackageArgs {
  public static final PackageArgs DEFAULT =
      PackageArgs.builder()
          .setDefaultVisibility(RuleVisibility.PRIVATE)
          .setDefaultTestOnly(false)
          .setFeatures(FeatureSet.EMPTY)
          .setLicense(License.NO_LICENSE)
          .setDistribs(License.DEFAULT_DISTRIB)
          .setDefaultCompatibleWith(ImmutableSet.of())
          .setDefaultRestrictedTo(ImmutableSet.of())
          .setDefaultPackageMetadata(ImmutableList.of())
          .build();

  /** See {@link Package#getDefaultVisibility()}. */
  @Nullable
  abstract RuleVisibility defaultVisibility();

  /** See {@link Package#getDefaultTestOnly()}. */
  @Nullable
  abstract Boolean defaultTestOnly();

  /** See {@link Package#getDefaultDeprecation()}. */
  @Nullable
  abstract String defaultDeprecation();

  /** See {@link Package#getFeatures()}. */
  abstract FeatureSet features();

  /** See {@link Package#getDefaultLicense()}. */
  @Nullable
  abstract License license();

  /** See {@link Package#getDefaultDistribs()}. */
  @Nullable
  abstract ImmutableSet<DistributionType> distribs();

  /** See {@link Package#getDefaultCompatibleWith()}. */
  @Nullable
  abstract ImmutableSet<Label> defaultCompatibleWith();

  /** See {@link Package#getDefaultRestrictedTo()}. */
  @Nullable
  abstract ImmutableSet<Label> defaultRestrictedTo();

  /** See {@link Package#getDefaultPackageMetadata()}. */
  @Nullable
  abstract ImmutableList<Label> defaultPackageMetadata();

  public static Builder builder() {
    return new AutoValue_PackageArgs.Builder().setFeatures(FeatureSet.EMPTY);
  }

  abstract Builder toBuilder();

  /** Builder type for {@link PackageArgs}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setDefaultVisibility(RuleVisibility x);

    public abstract Builder setDefaultTestOnly(Boolean x);

    public abstract Builder setDefaultDeprecation(String x);

    abstract FeatureSet features();

    public abstract Builder setFeatures(FeatureSet x);

    @CanIgnoreReturnValue
    public final Builder mergeFeatures(FeatureSet x) {
      return setFeatures(FeatureSet.merge(features(), x));
    }

    public abstract Builder setLicense(License x);

    public abstract Builder setDistribs(Set<DistributionType> x);

    /** Note that we don't check dupes in this method. Check beforehand! */
    public abstract Builder setDefaultCompatibleWith(Iterable<Label> x);

    /** Note that we don't check dupes in this method. Check beforehand! */
    public abstract Builder setDefaultRestrictedTo(Iterable<Label> x);

    @Nullable
    abstract ImmutableList<Label> defaultPackageMetadata();

    /** Note that we don't check dupes in this method. Check beforehand! */
    public abstract Builder setDefaultPackageMetadata(List<Label> x);

    public abstract PackageArgs build();
  }

  private static void throwIfHasDupes(List<Label> labels, String what) throws EvalException {
    var dupes = ImmutableSortedSet.copyOf(CollectionUtils.duplicatedElementsOf(labels));
    if (!dupes.isEmpty()) {
      throw Starlark.errorf("duplicate label(s) in %s: %s", what, Joiner.on(", ").join(dupes));
    }
  }

  /**
   * Processes the given Starlark parameter to the {@code package()/repo()} call into a field on a
   * {@link Builder} object.
   */
  public static void processParam(
      String name, Object rawValue, String what, LabelConverter labelConverter, Builder builder)
      throws EvalException {
    if (name.equals("default_visibility")) {
      List<Label> value = BuildType.LABEL_LIST.convert(rawValue, what, labelConverter);
      builder.setDefaultVisibility(RuleVisibility.parse(value));

    } else if (name.equals("default_testonly")) {
      Boolean value = Type.BOOLEAN.convert(rawValue, what, labelConverter);
      builder.setDefaultTestOnly(value);

    } else if (name.equals("default_deprecation")) {
      String value = Type.STRING.convert(rawValue, what, labelConverter);
      builder.setDefaultDeprecation(value);

    } else if (name.equals("features")) {
      List<String> value = Type.STRING_LIST.convert(rawValue, what, labelConverter);
      builder.mergeFeatures(FeatureSet.parse(value));

    } else if (name.equals("licenses")) {
      License value = BuildType.LICENSE.convert(rawValue, what, labelConverter);
      builder.setLicense(value);

    } else if (name.equals("distribs")) {
      Set<DistributionType> value = BuildType.DISTRIBUTIONS.convert(rawValue, what, labelConverter);
      builder.setDistribs(value);

    } else if (name.equals("default_compatible_with")) {
      List<Label> value = BuildType.LABEL_LIST.convert(rawValue, what, labelConverter);
      throwIfHasDupes(value, name);
      builder.setDefaultCompatibleWith(value);

    } else if (name.equals("default_restricted_to")) {
      List<Label> value = BuildType.LABEL_LIST.convert(rawValue, what, labelConverter);
      throwIfHasDupes(value, name);
      builder.setDefaultRestrictedTo(value);

    } else if (name.equals("default_applicable_licenses")
        || name.equals("default_package_metadata")) {
      List<Label> value = BuildType.LABEL_LIST.convert(rawValue, what, labelConverter);
      if (builder.defaultPackageMetadata() != null) {
        throw Starlark.errorf(
            "Can not set both default_package_metadata and default_applicable_licenses."
                + " Move all declarations to default_package_metadata.");
      }
      throwIfHasDupes(value, name);
      builder.setDefaultPackageMetadata(value);

    } else {
      throw Starlark.errorf("unexpected keyword argument: %s", name);
    }
  }

  /**
   * Returns a new {@link PackageArgs} containing the result of merging {@code other} into {@code
   * this}. {@code other}'s fields take precedence if specified.
   */
  public PackageArgs mergeWith(PackageArgs other) {
    Builder builder = toBuilder();
    if (other.defaultVisibility() != null) {
      builder.setDefaultVisibility(other.defaultVisibility());
    }
    if (other.defaultTestOnly() != null) {
      builder.setDefaultTestOnly(other.defaultTestOnly());
    }
    if (other.defaultDeprecation() != null) {
      builder.setDefaultDeprecation(other.defaultDeprecation());
    }
    if (!other.features().equals(FeatureSet.EMPTY)) {
      builder.mergeFeatures(other.features());
    }
    if (other.license() != null) {
      builder.setLicense(other.license());
    }
    if (other.distribs() != null) {
      builder.setDistribs(other.distribs());
    }
    if (other.defaultCompatibleWith() != null) {
      builder.setDefaultCompatibleWith(other.defaultCompatibleWith());
    }
    if (other.defaultRestrictedTo() != null) {
      builder.setDefaultRestrictedTo(other.defaultRestrictedTo());
    }
    if (other.defaultPackageMetadata() != null) {
      builder.setDefaultPackageMetadata(other.defaultPackageMetadata());
    }
    return builder.build();
  }
}
