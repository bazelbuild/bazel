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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ComparisonChain;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.apple.DottedVersionApi;
import java.util.ArrayList;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;

/**
 * Represents Xcode versions and allows parsing them.
 *
 * <p>Xcode versions are formed of multiple components, separated by periods, for example {@code
 * 4.5.6} or {@code 5.0.1beta2}. Components must start with a non-negative integer and at least one
 * component must be present.
 *
 * <p>Specifically, the format of a component is {@code \d+([a-z0-9]*?)?(\d+)?}.
 *
 * <p>If this smells a lot like semver, it does, but Xcode versions are sometimes special. This is
 * why this class is in the {@code apple} package and has to remain as such.
 *
 * <p>Dotted versions are ordered using natural integer sorting on components in order from first to
 * last where any missing element is considered to have the value 0 if they don't contain any
 * non-numeric characters. For example:
 *
 * <pre>
 *   3.1.25 > 3.1.1
 *   3.1.20 > 3.1.2
 *   3.1.1 > 3.1
 *   3.1 == 3.1.0.0
 *   3.2 > 3.1.8
 * </pre>
 *
 * <p>If the component contains any alphabetic characters after the leading integer, it is
 * considered <strong>smaller</strong> than any components with the same integer but larger than any
 * component with a smaller integer. If the integers are the same, the alphabetic sequences are
 * compared lexicographically, and if <i>they</i> turn out to be the same, the final (optional)
 * integer is compared. As with the leading integer, this final integer is considered to be 0 if not
 * present. For example:
 *
 * <pre>
 *   3.1.1 > 3.1.1beta3
 *   3.1.1beta1 > 3.1.0
 *   3.1 > 3.1.0alpha1
 *
 *   3.1.0beta0 > 3.1.0alpha5.6
 *   3.4.2alpha2 > 3.4.2alpha1
 *   3.4.2alpha2 > 3.4.2alpha1.5
 *   3.1alpha1 > 3.1alpha
 * </pre>
 *
 * <p>This class is immutable and can safely be shared among threads.
 */
@Immutable
@AutoCodec
public final class DottedVersion implements DottedVersionApi<DottedVersion> {
  /**
   * Wrapper class for {@link DottedVersion} whose {@link #equals(Object)} method is string
   * equality.
   *
   * <p>This is necessary because Bazel assumes that {@link
   * com.google.devtools.build.lib.analysis.config.FragmentOptions} that are equal yield fragments
   * that are the same. However, this does not hold if the options hold a {@link DottedVersion}
   * because trailing zeroes are not considered significant when comparing them, but they do matter
   * in configuration fragments (for example, they end up in output directory names).
   *
   * <p>When read from the {@code settings} dictionary in a Starlark transition function, these
   * values are effectively opaque and need to be converted to strings for further use, such as
   * comparing them by passing the string to {@code apple_common.dotted_version} to construct an
   * instance of the actual version object.
   */
  @Immutable
  public static final class Option implements StarlarkValue {
    private final DottedVersion version;

    private Option(DottedVersion version) {
      this.version = Preconditions.checkNotNull(version);
    }

    public DottedVersion get() {
      return version;
    }

    @Override
    public boolean isImmutable() {
      return true;
    }

    @Override
    public void repr(Printer printer) {
      printer.append(version.toString());
    }

    @Override
    public String toString() {
      return version.toString();
    }

    @Override
    public int hashCode() {
      return version.stringRepresentation.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }

      if (!(o instanceof Option)) {
        return false;
      }

      return version.stringRepresentation.equals(((Option) o).version.stringRepresentation);
    }
  }

  public static DottedVersion maybeUnwrap(DottedVersion.Option option) {
    return option != null ? option.get() : null;
  }

  public static Option option(DottedVersion version) {
    return version == null ? null : new Option(version);
  }
  private static final Splitter DOT_SPLITTER = Splitter.on('.');
  private static final Pattern COMPONENT_PATTERN =
      Pattern.compile("(\\d+)([a-z0-9]*?)?(\\d+)?", Pattern.CASE_INSENSITIVE);
  private static final Pattern DESCRIPTIVE_COMPONENT_PATTERN =
      Pattern.compile("([a-z]\\w*)", Pattern.CASE_INSENSITIVE);
  private static final String ILLEGAL_VERSION =
      "Dotted version components must all start with the form \\d+([a-z0-9]*?)?(\\d+)? "
          + "but got '%s'";
  private static final String NO_ALPHA_SEQUENCE = null;
  private static final Component ZERO_COMPONENT = new Component(0, NO_ALPHA_SEQUENCE, 0, "0");

  /** Exception thrown when parsing an invalid dotted version. */
  public static class InvalidDottedVersionException extends Exception {
    InvalidDottedVersionException(String msg) {
      super(msg);
    }

    InvalidDottedVersionException(String msg, Throwable cause) {
      super(msg, cause);
    }
  }

  /**
   * Create a dotted version by parsing the given version string. Throws an unchecked exception if
   * the argument is malformed.
   */
  public static DottedVersion fromStringUnchecked(String version) {
    try {
      return fromString(version);
    } catch (InvalidDottedVersionException e) {
      throw new IllegalArgumentException(e);
    }
  }

  /**
   * Generates a new dotted version from the given version string.
   *
   * @throws InvalidDottedVersionException if the passed string is not a valid dotted version
   */
  public static DottedVersion fromString(String version) throws InvalidDottedVersionException {
    if (Strings.isNullOrEmpty(version)) {
      throw new InvalidDottedVersionException(String.format(ILLEGAL_VERSION, version));
    }
    ArrayList<Component> components = new ArrayList<>();
    for (String component : DOT_SPLITTER.split(version)) {
      if (isDescriptiveComponent(component)) {
        break;
      }
      components.add(toComponent(component, version));
    }

    if (components.isEmpty()) {
      throw new InvalidDottedVersionException(String.format(ILLEGAL_VERSION, version));
    }

    int numOriginalComponents = components.size();

    // Remove trailing (but not the first or middle) zero components for easier comparison and
    // hashcoding.
    for (int i = components.size() - 1; i > 0; i--) {
      if (components.get(i).equals(ZERO_COMPONENT)) {
        components.remove(i);
      } else {
        break;
      }
    }

    return new DottedVersion(ImmutableList.copyOf(components), version, numOriginalComponents);
  }

  // Some of special build versions contains descriptive components like "experimental" or
  // "internal". These components are usually by the end of version number, and can be ignored.
  private static boolean isDescriptiveComponent(String component) {
    return DESCRIPTIVE_COMPONENT_PATTERN.matcher(component).matches();
  }

  private static Component toComponent(String component, String version)
      throws InvalidDottedVersionException {
    Matcher parsedComponent = COMPONENT_PATTERN.matcher(component);
    if (!parsedComponent.matches()) {
      throw new InvalidDottedVersionException(String.format(ILLEGAL_VERSION, version));
    }

    int firstNumber;
    String alphaSequence = NO_ALPHA_SEQUENCE;
    int secondNumber = 0;
    firstNumber = parseNumber(parsedComponent, 1, version);

    if (!Strings.isNullOrEmpty(parsedComponent.group(2))) {
      alphaSequence = parsedComponent.group(2);
    }

    if (!Strings.isNullOrEmpty(parsedComponent.group(3))) {
      secondNumber = parseNumber(parsedComponent, 3, version);
    }

    return new Component(firstNumber, alphaSequence, secondNumber, component);
  }

  private static int parseNumber(Matcher parsedComponent, int group, String version)
      throws InvalidDottedVersionException {
    int firstNumber;
    try {
      firstNumber = Integer.parseInt(parsedComponent.group(group));
    } catch (NumberFormatException e) {
      throw new InvalidDottedVersionException(String.format(ILLEGAL_VERSION, version), e);
    }
    return firstNumber;
  }

  private final ImmutableList<Component> components;
  private final String stringRepresentation;
  private final int numOriginalComponents;

  @AutoCodec.VisibleForSerialization
  DottedVersion(
      ImmutableList<Component> components, String stringRepresentation, int numOriginalComponents) {
    this.components = components;
    this.stringRepresentation = stringRepresentation;
    this.numOriginalComponents = numOriginalComponents;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public int compareTo(DottedVersion other) {
    int maxComponents = Math.max(components.size(), other.components.size());
    for (int componentIndex = 0; componentIndex < maxComponents; componentIndex++) {
      Component myComponent = getComponent(componentIndex);
      Component otherComponent = other.getComponent(componentIndex);
      int comparison = myComponent.compareTo(otherComponent);
      if (comparison != 0) {
        return comparison;
      }
    }
    return 0;
  }

  @Override
  public int compareTo_starlark(DottedVersion other) {
    return compareTo(other);
  }

  /**
   * Returns the string representation of this dotted version, padded or truncated to the specified
   * number of components.
   *
   * <p>For example, a dotted version of "7.3.0" will return "7" if one is requested, "7.3" if two
   * are requested, "7.3.0" if three are requested, and "7.3.0.0" if four are requested.
   *
   * @param numComponents a positive number of dot-separated numbers that should be present in the
   *     returned string representation
   */
  public String toStringWithComponents(int numComponents) {
    Preconditions.checkArgument(numComponents > 0,
        "Can't serialize as a version with %s components", numComponents);
    ImmutableList.Builder<Component> stringComponents = ImmutableList.builder();
    if (numComponents <= components.size()) {
      stringComponents.addAll(components.subList(0, numComponents));
    } else {
      stringComponents.addAll(components);
      for (int i = components.size(); i < numComponents; i++) {
        stringComponents.add(ZERO_COMPONENT);
      }
    }
    return Joiner.on('.').join(stringComponents.build());
  }

  /**
   * Returns the string representation of this dotted version, padded to a minimum number of
   * components if the string representation does not already contain that many components.
   *
   * <p>For example, a dotted version of "7.3" will return "7.3" with either one or two components
   * requested, "7.3.0" if three are requested, and "7.3.0.0" if four are requested.
   *
   * <p>Trailing zero components at the end of a string representation will not be removed. For
   * example, a dotted version of "1.0.0" will return "1.0.0" if only one or two components are
   * requested.
   *
   * @param numMinComponents the minimum number of dot-separated numbers that should be present in
   *     the returned string representation
   */
  public String toStringWithMinimumComponents(int numMinComponents) {
    return toStringWithComponents(Math.max(this.numOriginalComponents, numMinComponents));
  }

  /**
   * Returns true if this version number has any alphabetic characters, such as 'alpha' in
   * "7.3alpha.2".
   */
  public boolean hasAlphabeticCharacters() {
    for (Component component : components) {
      if (!Objects.equals(component.alphaSequence, NO_ALPHA_SEQUENCE)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the number of components in this version number. For example, "7.3.0" has three
   * components.
   */
  public int numComponents() {
    return components.size();
  }

  @Override
  public String toString() {
    return stringRepresentation;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (other == null || getClass() != other.getClass()) {
      return false;
    }

    return compareTo((DottedVersion) other) == 0;
  }

  @Override
  public int hashCode() {
    return Objects.hash(components);
  }

  private Component getComponent(int groupIndex) {
    if (components.size() > groupIndex) {
      return components.get(groupIndex);
    }
    return ZERO_COMPONENT;
  }

  @Override
  public void repr(Printer printer) {
    printer.append(stringRepresentation);
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static final class Component implements Comparable<Component> {
    private final int firstNumber;
    @Nullable private final String alphaSequence;
    private final int secondNumber;
    private final String stringRepresentation;

    @AutoCodec.VisibleForSerialization
    Component(
        int firstNumber,
        @Nullable String alphaSequence,
        int secondNumber,
        String stringRepresentation) {
      this.firstNumber = firstNumber;
      this.alphaSequence = alphaSequence;
      this.secondNumber = secondNumber;
      this.stringRepresentation = stringRepresentation;
    }

    @Override
    public int compareTo(Component other) {
      return ComparisonChain.start()
          .compare(firstNumber, other.firstNumber)
          .compare(alphaSequence, other.alphaSequence, Ordering.natural().nullsLast())
          .compare(secondNumber, other.secondNumber)
          .result();
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }
      if (other == null || getClass() != other.getClass()) {
        return false;
      }

      return compareTo((Component) other) == 0;
    }

    @Override
    public int hashCode() {
      return Objects.hash(firstNumber, alphaSequence, secondNumber);
    }

    @Override
    public String toString() {
      return stringRepresentation;
    }
  }
}
