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
 *
 * <p>Unless otherwise specified, these are only used when the rule does not provide an explicit
 * override value in the associated attribute.
 */
@AutoValue
public abstract class PackageArgs {
  public static final PackageArgs EMPTY = PackageArgs.builder().build();

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

  /** The default visibility value for the package. */
  @Nullable
  public abstract RuleVisibility defaultVisibility();

  /** The default testonly value for the package. */
  @Nullable
  public abstract Boolean defaultTestOnly();

  /** The default deprecation value for the package. */
  @Nullable
  public abstract String defaultDeprecation();

  /**
   * The default (generally C/C++) features value for the package.
   *
   * <p>Note that this is actually additive with features set by a rule where the rule has priority
   * for turning specific features on or off.
   */
  public abstract FeatureSet features();

  /** The default license value for the package. */
  @Nullable
  public abstract License license();

  /** The default distributions value for the package. */
  @Nullable
  public abstract ImmutableSet<DistributionType> distribs();

  /** The default {@link RuleClass#COMPATIBLE_ENVIRONMENT_ATTR} value for the package. */
  @Nullable
  public abstract ImmutableSet<Label> defaultCompatibleWith();

  /** The default {@link RuleClass#RESTRICTED_ENVIRONMENT_ATTR} value for the package. */
  @Nullable
  public abstract ImmutableSet<Label> defaultRestrictedTo();

  /** The default package metadata list value for the package. */
  @Nullable
  public abstract ImmutableList<Label> defaultPackageMetadata();

  // TODO(blaze-team): this should just act like other attributes in that
  //   it is public and does not have getters defined
  /** The default (C/C++) header strictness checking mode for the package. */
  @Nullable
  abstract String defaultHdrsCheck();

  /** Gets the default header checking mode. */
  public String getDefaultHdrsCheck() {
    return defaultHdrsCheck() != null ? defaultHdrsCheck() : "strict";
  }

  /** Returns whether the default header checking mode has been set or it is the default value. */
  public boolean isDefaultHdrsCheckSet() {
    return defaultHdrsCheck() != null;
  }

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

    public abstract Builder setDefaultHdrsCheck(String x);

    public abstract PackageArgs build();
  }

  private static List<Label> throwIfHasDupes(List<Label> labels, String what) throws EvalException {
    var dupes = ImmutableSortedSet.copyOf(CollectionUtils.duplicatedElementsOf(labels));
    if (!dupes.isEmpty()) {
      throw Starlark.errorf("duplicate label(s) in %s: %s", what, Joiner.on(", ").join(dupes));
    }
    return labels;
  }

  /**
   * Processes the given Starlark parameter to the {@code package()/repo()} call into a field on a
   * {@link Builder} object.
   */
  public static void processParam(
      String name, Object rawValue, String what, LabelConverter labelConverter, Builder builder)
      throws EvalException {
    switch (name) {
      case "default_visibility":
        builder.setDefaultVisibility(
            RuleVisibility.parse(BuildType.LABEL_LIST.convert(rawValue, what, labelConverter)));
        break;
      case "default_testonly":
        builder.setDefaultTestOnly(Type.BOOLEAN.convert(rawValue, what, labelConverter));
        break;
      case "default_deprecation":
        builder.setDefaultDeprecation(Type.STRING.convert(rawValue, what, labelConverter));
        break;
      case "features":
        builder.mergeFeatures(
            FeatureSet.parse(Type.STRING_LIST.convert(rawValue, what, labelConverter)));
        break;
      case "licenses":
        builder.setLicense(BuildType.LICENSE.convert(rawValue, what, labelConverter));
        break;
      case "distribs":
        builder.setDistribs(BuildType.DISTRIBUTIONS.convert(rawValue, what, labelConverter));
        break;
      case "default_compatible_with":
        builder.setDefaultCompatibleWith(
            throwIfHasDupes(BuildType.LABEL_LIST.convert(rawValue, what, labelConverter), name));
        break;
      case "default_restricted_to":
        builder.setDefaultRestrictedTo(
            throwIfHasDupes(BuildType.LABEL_LIST.convert(rawValue, what, labelConverter), name));
        break;
      case "default_applicable_licenses":
      case "default_package_metadata":
        if (builder.defaultPackageMetadata() != null) {
          throw Starlark.errorf(
              "Can not set both default_package_metadata and default_applicable_licenses."
                  + " Move all declarations to default_package_metadata.");
        }
        builder.setDefaultPackageMetadata(
            throwIfHasDupes(BuildType.LABEL_LIST.convert(rawValue, what, labelConverter), name));
        break;
      case "default_hdrs_check":
        builder.setDefaultHdrsCheck(Type.STRING.convert(rawValue, what, labelConverter));
        break;
      default:
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
    if (other.defaultHdrsCheck() != null) {
      builder.setDefaultHdrsCheck(other.defaultHdrsCheck());
    }
    return builder.build();
  }
}
