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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * A RuleVisibility specifies which other rules can depend on a specified rule. Note that the actual
 * method that performs this check is declared in RuleConfiguredTargetVisibility.
 *
 * <p>The conversion to ConfiguredTargetVisibility is handled in an ugly if-ladder, because I want
 * to avoid this package depending on build.lib.view.
 *
 * <p>All implementations of this interface are immutable.
 */
public interface RuleVisibility {

  /**
   * Returns the list of all labels comprising this visibility.
   *
   * <p>This includes labels that are not loadable, such as //visibility:public and //foo:__pkg__.
   */
  List<Label> getDeclaredLabels();

  /**
   * Same as {@link #getDeclaredLabels}, but excludes labels that cannot be loaded.
   *
   * <p>I.e., this returns the labels of the top-level {@code package_group}s that must be loaded in
   * order to determine the complete set of packages represented by this visibility. (Additional
   * {@code package_group}s may need to be loaded due to their {@code includes} attribute.)
   */
  List<Label> getDependencyLabels();

  @SerializationConstant Label PUBLIC_LABEL = Label.parseCanonicalUnchecked("//visibility:public");

  @SerializationConstant
  Label PRIVATE_LABEL = Label.parseCanonicalUnchecked("//visibility:private");

  // Constant for memory efficiency; see b/370873477.
  @SerializationConstant
  ImmutableList<Label> PUBLIC_DECLARED_LABELS = ImmutableList.of(PUBLIC_LABEL);

  // Constant for memory efficiency; see b/370873477.
  @SerializationConstant
  ImmutableList<Label> PRIVATE_DECLARED_LABELS = ImmutableList.of(PRIVATE_LABEL);

  @SerializationConstant
  RuleVisibility PUBLIC =
      new RuleVisibility() {
        @Override
        public ImmutableList<Label> getDeclaredLabels() {
          return PUBLIC_DECLARED_LABELS;
        }

        @Override
        public ImmutableList<Label> getDependencyLabels() {
          return ImmutableList.of();
        }

        @Override
        public String toString() {
          return PUBLIC_LABEL.toString();
        }
      };

  @SerializationConstant
  RuleVisibility PRIVATE =
      new RuleVisibility() {
        @Override
        public ImmutableList<Label> getDeclaredLabels() {
          return PRIVATE_DECLARED_LABELS;
        }

        @Override
        public ImmutableList<Label> getDependencyLabels() {
          return ImmutableList.of();
        }

        @Override
        public String toString() {
          return PRIVATE_LABEL.toString();
        }
      };

  /** Validates and parses the given labels into a {@link RuleVisibility}. */
  static RuleVisibility parse(List<Label> labels) throws EvalException {
    return parseUnchecked(validateAndSimplify(labels));
  }

  /**
   * Same as {@link #parse} except does not perform validation checks or public/private
   * simplification.
   *
   * <p>Use only after the given labels have been {@linkplain #validateAndSimplify validated and
   * simplified}.
   */
  static RuleVisibility parseUnchecked(List<Label> labels) {
    RuleVisibility result = parseIfConstant(labels);
    if (result != null) {
      return result;
    }
    return PackageGroupsRuleVisibility.create(labels);
  }

  /**
   * If the given list of labels represents a constant {@link RuleVisibility} ({@link #PUBLIC} or
   * {@link #PRIVATE}), returns that visibility instance, otherwise returns {@code null}.
   *
   * <p>Use only after the given labels have been {@linkplain #validateAndSimplify validated and
   * simplified}.
   */
  @Nullable
  static RuleVisibility parseIfConstant(List<Label> labels) {
    if (labels.size() != 1) {
      return null;
    }
    Label label = labels.getFirst();
    if (label.equals(PUBLIC_LABEL)) {
      return PUBLIC;
    }
    if (label.equals(PRIVATE_LABEL)) {
      return PRIVATE;
    }
    return null;
  }

  @CanIgnoreReturnValue
  private static Label validate(Label label) throws EvalException {
    if (label.getPackageIdentifier().equals(PUBLIC_LABEL.getPackageIdentifier())
        && PackageSpecification.fromLabel(label) == null) {
      // In other words, if the label is in //visibility and is not //visibility:public,
      // //visibility:private, or (for the unusual case where //visibility exists as a package)
      // //visibility:__pkg__ or //visibility:__subpackages__
      throw Starlark.errorf(
          "Invalid visibility label '%s'; did you mean //visibility:public or"
              + " //visibility:private?",
          label);
    }
    return label;
  }

  /**
   * Validates visibility labels, simplifies a list containing "//visibility:public" to
   * ["//visibility:public"], drops "//visibility:private" if it occurs with other labels, and
   * canonicalizes an empty list to ["//visibility:private"].
   *
   * @param labels list of visibility labels; not modified even if mutable.
   * @return either {@code labels} unmodified if it does not require simplification, or a new
   *     simplified list of visibility labels.
   */
  // TODO(arostovtsev): we ought to uniquify the labels, matching the behavior of {@link
  // #concatWithElement}; note that this would be an incompatible change (affects query output).
  static List<Label> validateAndSimplify(List<Label> labels) throws EvalException {
    boolean hasPublicLabel = false;
    int numPrivateLabels = 0;
    for (Label label : labels) {
      if (label.equals(PUBLIC_LABEL)) {
        // Do not short-circuit here; we want to validate all the labels.
        hasPublicLabel = true;
      } else if (label.equals(PRIVATE_LABEL)) {
        numPrivateLabels++;
      } else {
        validate(label);
      }
    }
    if (hasPublicLabel) {
      return PUBLIC_DECLARED_LABELS;
    }
    if (numPrivateLabels == labels.size()) {
      return PRIVATE_DECLARED_LABELS;
    }
    if (numPrivateLabels == 0) {
      return labels;
    }
    ImmutableList.Builder<Label> withoutPrivateLabels =
        ImmutableList.builderWithExpectedSize(labels.size() - numPrivateLabels);
    for (Label label : labels) {
      if (!label.equals(PRIVATE_LABEL)) {
        withoutPrivateLabels.add(label);
      }
    }
    return withoutPrivateLabels.build();
  }

  /**
   * Returns a {@code RuleVisibility} representing the logical result of concatenating the given
   * {@code visibility} with the additional {@code element}.
   *
   * <p>If {@code element} or {@code visibility} is public, the result is public. If {@code element}
   * or {@code visibility} is private, the result is {@code visibility} or a visibility consisting
   * solely of {@code element}, respectively.
   *
   * <p>If {@code element} is already present in {@code visibility}, the result is just {@code
   * visibility}.
   *
   * @throws EvalException if there's a problem parsing {@code element} into a visibility
   */
  static RuleVisibility concatWithElement(RuleVisibility visibility, Label element)
      throws EvalException {
    if (visibility.equals(RuleVisibility.PRIVATE)) {
      // Left-side private is dropped.
      return parse(ImmutableList.of(element));
    } else if (element.equals(PRIVATE_LABEL)) {
      // Right-side private is dropped.
      return visibility;
    } else if (visibility.equals(RuleVisibility.PUBLIC) || element.equals(PUBLIC_LABEL)) {
      // Public is idempotent.
      return RuleVisibility.PUBLIC;
    } else {
      List<Label> items = visibility.getDeclaredLabels();
      if (items.contains(element)) {
        return visibility;
      } else {
        ImmutableList.Builder<Label> newItems = new ImmutableList.Builder<>();
        newItems.addAll(items);
        newItems.add(validate(element));
        return parseUnchecked(newItems.build());
      }
    }
  }

  /**
   * Convenience wrapper for {@link #concatWithElement} where the added element is the given
   * package.
   *
   * <p>Unlike that method, this does not throw EvalException.
   */
  static RuleVisibility concatWithPackage(
      RuleVisibility visibility, PackageIdentifier packageIdentifier) {
    Label pkgItem = Label.createUnvalidated(packageIdentifier, "__pkg__");
    try {
      return concatWithElement(visibility, pkgItem);
    } catch (EvalException ex) {
      throw new AssertionError(
          String.format("Unreachable; couldn't parse %s as visibility", pkgItem), ex);
    }
  }
}
