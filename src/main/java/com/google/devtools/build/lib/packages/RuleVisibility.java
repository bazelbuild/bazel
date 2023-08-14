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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
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
   * Returns the list of labels that need to be loaded so that the visibility decision can be made
   * during analysis time. E.g. for package group visibility, this is the list of package groups
   * referenced. Does not include labels that have special meanings in the visibility declaration,
   * e.g. "//visibility:*" or "//*:__pkg__".
   */
  List<Label> getDependencyLabels();

  /**
   * Returns the list of labels used during the declaration of this visibility. These do not
   * necessarily represent loadable labels: for example, for public or private visibilities, the
   * special labels "//visibility:*" will be returned, and so will be the special "//*:__pkg__"
   * labels indicating a single package.
   */
  List<Label> getDeclaredLabels();

  @SerializationConstant Label PUBLIC_LABEL = Label.parseCanonicalUnchecked("//visibility:public");

  @SerializationConstant
  Label PRIVATE_LABEL = Label.parseCanonicalUnchecked("//visibility:private");

  @SerializationConstant
  RuleVisibility PUBLIC =
      new RuleVisibility() {
        @Override
        public ImmutableList<Label> getDependencyLabels() {
          return ImmutableList.of();
        }

        @Override
        public ImmutableList<Label> getDeclaredLabels() {
          return ImmutableList.of(PUBLIC_LABEL);
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
        public ImmutableList<Label> getDependencyLabels() {
          return ImmutableList.of();
        }

        @Override
        public ImmutableList<Label> getDeclaredLabels() {
          return ImmutableList.of(PRIVATE_LABEL);
        }

        @Override
        public String toString() {
          return PRIVATE_LABEL.toString();
        }
      };

  /** Validates and parses the given labels into a {@link RuleVisibility}. */
  static RuleVisibility parse(List<Label> labels) throws EvalException {
    validate(labels);
    return parseUnchecked(labels);
  }

  /**
   * Same as {@link #parse} except does not perform validation checks.
   *
   * <p>Use only after the given labels have been {@linkplain #validate validated}.
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
   * <p>Use only after the given labels have been {@linkplain #validate validated}.
   */
  @Nullable
  static RuleVisibility parseIfConstant(List<Label> labels) {
    if (labels.size() != 1) {
      return null;
    }
    Label label = labels.get(0);
    if (label.equals(PUBLIC_LABEL)) {
      return PUBLIC;
    }
    if (label.equals(PRIVATE_LABEL)) {
      return PRIVATE;
    }
    return null;
  }

  static void validate(List<Label> labels) throws EvalException {
    if (labels.size() <= 1) {
      return;
    }
    for (Label label : labels) {
      if (label.equals(PUBLIC_LABEL) || label.equals(PRIVATE_LABEL)) {
        throw Starlark.errorf(
            "Public or private visibility labels (e.g. //visibility:public or"
                + " //visibility:private) cannot be used in combination with other labels");
      }
    }
  }
}

