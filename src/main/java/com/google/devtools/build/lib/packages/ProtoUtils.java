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

import static com.google.devtools.build.lib.packages.BuildType.DISTRIBUTIONS;
import static com.google.devtools.build.lib.packages.BuildType.FILESET_ENTRY_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_DICT_UNARY;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.OUTPUT;
import static com.google.devtools.build.lib.packages.BuildType.OUTPUT_LIST;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.INTEGER;
import static com.google.devtools.build.lib.syntax.Type.INTEGER_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_DICT;
import static com.google.devtools.build.lib.syntax.Type.STRING_DICT_UNARY;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST_DICT;

import com.google.common.base.Optional;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.Attribute.Discriminator;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Set;

/**
 * Shared code used in proto buffer output for rules and rule classes.
 */
public class ProtoUtils {
  /**
   * This map contains all attribute types that are recognized by the protocol
   * output formatter.
   *
   * <p>If you modify this map, please ensure that {@link #getTypeFromDiscriminator} can still
   * resolve a {@link Discriminator} value to exactly one {@link Type} (using an optional nodep
   * hint, as described below).
   */
  static final ImmutableMap<Type<?>, Discriminator> TYPE_MAP =
      new ImmutableMap.Builder<Type<?>, Discriminator>()
          .put(INTEGER, Discriminator.INTEGER)
          .put(DISTRIBUTIONS, Discriminator.DISTRIBUTION_SET)
          .put(LABEL, Discriminator.LABEL)
          // NODEP_LABEL attributes are not really strings. This is implemented
          // this way for the sake of backward compatibility.
          .put(NODEP_LABEL, Discriminator.STRING)
          .put(LABEL_LIST, Discriminator.LABEL_LIST)
          .put(NODEP_LABEL_LIST, Discriminator.STRING_LIST)
          .put(STRING, Discriminator.STRING)
          .put(STRING_LIST, Discriminator.STRING_LIST)
          .put(OUTPUT, Discriminator.OUTPUT)
          .put(OUTPUT_LIST, Discriminator.OUTPUT_LIST)
          .put(LICENSE, Discriminator.LICENSE)
          .put(STRING_DICT, Discriminator.STRING_DICT)
          .put(FILESET_ENTRY_LIST, Discriminator.FILESET_ENTRY_LIST)
          .put(LABEL_DICT_UNARY, Discriminator.LABEL_DICT_UNARY)
          .put(STRING_LIST_DICT, Discriminator.STRING_LIST_DICT)
          .put(BOOLEAN, Discriminator.BOOLEAN)
          .put(TRISTATE, Discriminator.TRISTATE)
          .put(INTEGER_LIST, Discriminator.INTEGER_LIST)
          .put(STRING_DICT_UNARY, Discriminator.STRING_DICT_UNARY)
          .build();

  static final ImmutableSet<Type<?>> NODEP_TYPES = ImmutableSet.of(NODEP_LABEL, NODEP_LABEL_LIST);

  static final ImmutableSetMultimap<Discriminator, Type<?>> INVERSE_TYPE_MAP =
      TYPE_MAP.asMultimap().inverse();

  /** Returns the {@link Discriminator} value corresponding to the provided {@link Type}. */
  public static Discriminator getDiscriminatorFromType(Type<?> type) {
    Preconditions.checkArgument(TYPE_MAP.containsKey(type), type);
    return TYPE_MAP.get(type);
  }

  /** Returns the {@link Type} associated with an {@link Build.Attribute}. */
  static Type<?> getTypeFromAttributePb(
      Build.Attribute attrPb, String ruleClassName, String attrName) {
    Optional<Boolean> nodepHint =
        attrPb.hasNodep() ? Optional.of(attrPb.getNodep()) : Optional.<Boolean>absent();
    Discriminator attrPbDiscriminator = attrPb.getType();
    boolean isSelectorList = attrPbDiscriminator.equals(Discriminator.SELECTOR_LIST);
    return getTypeFromDiscriminator(
        isSelectorList ? attrPb.getSelectorList().getType() : attrPbDiscriminator,
        nodepHint,
        ruleClassName,
        attrName);
  }

  /**
   * Returns the set of {@link Type}s associated with a {@link Discriminator} value.
   *
   * <p>The set will contain more than one {@link Type} when {@param discriminator} is either
   * {@link Discriminator#STRING} or {@link Discriminator#STRING_LIST}, because each of them
   * corresponds with two {@link Type} values. A nodeps hint is needed to determine which {@link
   * Type} applies.
   */
  private static ImmutableSet<Type<?>> getTypesFromDiscriminator(Discriminator discriminator) {
    Preconditions.checkArgument(INVERSE_TYPE_MAP.containsKey(discriminator), discriminator);
    return INVERSE_TYPE_MAP.get(discriminator);
  }

  /**
   * Returns the {@link Type} associated with a {@link Discriminator} value, given an optional
   * nodeps hint.
   */
  private static Type<?> getTypeFromDiscriminator(
      Discriminator discriminator,
      Optional<Boolean> nodeps,
      String ruleClassName,
      String attrName) {
    Preconditions.checkArgument(INVERSE_TYPE_MAP.containsKey(discriminator), discriminator);
    ImmutableSet<Type<?>> possibleTypes = ProtoUtils.getTypesFromDiscriminator(discriminator);
    Type<?> preciseType;
    if (possibleTypes.size() == 1) {
      preciseType = Iterables.getOnlyElement(possibleTypes);
    } else {
      // If there is more than one possible type associated with the discriminator, then the
      // discriminator must be either Discriminator.STRING or Discriminator.STRING_LIST.
      //
      // If it is Discriminator.STRING, then its possible Type<?>s are {NODEP_LABEL, STRING}. The
      // nodeps hint must be present in order to distinguish between them. If nodeps is true,
      // then the Type<?> must be NODEP_LABEL, and if false, it must be STRING.
      //
      // A similar relation holds for the Discriminator value STRING_LIST, and its possible
      // Type<?>s {NODEP_LABEL_LIST, STRING_LIST}.

      Preconditions.checkArgument(nodeps.isPresent(),
          "Nodeps hint is required when discriminator is associated with more than one type."
              + " Discriminator: \"%s\", Rule class: \"%s\", Attr: \"%s\"", discriminator,
          ruleClassName, attrName);
      if (nodeps.get()) {
        Set<Type<?>> nodepType = Sets.filter(possibleTypes, Predicates.in(NODEP_TYPES));
        Preconditions.checkState(nodepType.size() == 1,
            "There should be exactly one NODEP type associated with discriminator \"%s\""
                + ", but found these: %s. Rule class: \"%s\", Attr: \"%s\"", discriminator,
            nodepType, ruleClassName, attrName);
        preciseType = Iterables.getOnlyElement(nodepType);
      } else {
        Set<Type<?>> notNodepType =
            Sets.filter(possibleTypes, Predicates.not(Predicates.in(NODEP_TYPES)));
        Preconditions.checkState(notNodepType.size() == 1,
            "There should be exactly one non-NODEP type associated with discriminator \"%s\""
                + ", but found these: %s. Rule class: \"%s\", Attr: \"%s\"", discriminator,
            notNodepType, ruleClassName, attrName);
        preciseType = Iterables.getOnlyElement(notNodepType);
      }
    }
    return preciseType;
  }
}
