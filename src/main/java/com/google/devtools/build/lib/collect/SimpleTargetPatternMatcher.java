// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A simple matcher for checking whether a given label is part is a set of simple target patterns.
 *
 * <p>This does not implement full target patterns. Specifically, it handles:
 *
 * <ul>
 *   <li>Absolute labels, such as {@code //package:target} or {@code //package/subpackage}
 *   <li>Absolute package paths and all subpackages and targets, such as {@code //package/...}
 *   <li>Negative patterns of the above, such as {@code -//package:target}
 * </ul>
 *
 * It does not handle:
 *
 * <ul>
 *   <li>Relative labels (all labels with no repository are assumed to be in the main repository)
 *   <li>The {@code :all} or {@code :*} qualifiers
 * </ul>
 *
 * Patterns are processed in the order given, including negative patterns that override previous
 * patterns. This means that if the patterns are
 *
 * <ul>
 *   <li>{@code //package/...}
 *   <li>{@code -//package/subpackage/...}
 *   <li>{@code //package/subpackage/further/...}
 * </ul>
 *
 * then the labels {@code //package:something}, {@code //package/another} and {@code
 * //package/subpackage/further:anything} all match, but the label {@code
 * //package/subpackage:something} does not match.
 *
 * <p>Further note that this class does no loading of BUILD files and performs no verification that
 * targets actually exist: it simply matches abstract labels against patterns.
 */
public class SimpleTargetPatternMatcher {
  public static SimpleTargetPatternMatcher create(ImmutableList<String> patterns)
      throws LabelSyntaxException {
    ImmutableList.Builder<SinglePatternMatcher> singlePatternMatcherBuilder =
        ImmutableList.builder();
    for (String pattern : patterns) {
      SinglePatternMatcher matcher = SimpleTargetPatternMatcher.createSinglePatternMatcher(pattern);
      singlePatternMatcherBuilder.add(matcher);
    }
    return new SimpleTargetPatternMatcher(singlePatternMatcherBuilder.build());
  }

  private final ImmutableList<SinglePatternMatcher> singlePatternMatchers;

  private SimpleTargetPatternMatcher(ImmutableList<SinglePatternMatcher> singlePatternMatchers) {
    this.singlePatternMatchers = singlePatternMatchers;
  }

  public boolean isEmpty() {
    return this.singlePatternMatchers.isEmpty();
  }

  /** Returns {@code true} if the label matches all patterns in this matcher. */
  public boolean contains(Label label) {
    if (this.singlePatternMatchers.isEmpty()) {
      return false;
    }

    // Check each sub-matcher.
    MatchResult result = MatchResult.EXCLUDE;
    for (SinglePatternMatcher matcher : this.singlePatternMatchers) {
      MatchResult matchResult = matcher.matches(label);
      if (matchResult == MatchResult.INCLUDE || matchResult == MatchResult.EXCLUDE) {
        result = matchResult;
      }
    }
    return result == MatchResult.INCLUDE;
  }

  @Override
  public String toString() {
    String joined =
        this.singlePatternMatchers.stream()
            .map(SinglePatternMatcher::toString)
            .collect(joining(","));
    return String.format("[%s]", joined);
  }

  @Override
  public boolean equals(Object other) {
    if (other instanceof SimpleTargetPatternMatcher otherMatcher) {
      return this.singlePatternMatchers.equals(otherMatcher.singlePatternMatchers);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return this.singlePatternMatchers.hashCode();
  }

  private static SinglePatternMatcher createSinglePatternMatcher(String pattern)
      throws LabelSyntaxException {
    if (pattern.startsWith("-")) {
      // Strip off the leading '-' and create a matcher for what remains. This will technically
      // handle a series of nested negative patterns (like `---//exact:target`), but isn't worth
      // detecting and throwing an error.
      pattern = pattern.substring(1);
      SinglePatternMatcher inner = createSinglePatternMatcher(pattern);
      return new NegativeMatcher(inner);
    } else if (pattern.endsWith("/...")) {
      return new WildcardMatcher(pattern);
    }

    // Just match the pattern as an exact label.
    return new ExactMatcher(pattern);
  }

  private enum MatchResult {
    INCLUDE,
    EXCLUDE,
    NOT_RELEVANT;
  }

  private sealed interface SinglePatternMatcher
      permits ExactMatcher, NegativeMatcher, WildcardMatcher {
    MatchResult matches(Label label);
  }

  /** Checks if the given label exactly matches the pattern. */
  private static final class ExactMatcher implements SinglePatternMatcher {
    private final String rawPattern;
    private final Label label;

    private ExactMatcher(String pattern) throws LabelSyntaxException {
      this.rawPattern = pattern;
      this.label = Label.parseCanonical(pattern);
    }

    @Override
    public MatchResult matches(Label label) {
      if (this.label.equals(label)) {
        return MatchResult.INCLUDE;
      }
      return MatchResult.NOT_RELEVANT;
    }

    @Override
    public String toString() {
      return this.rawPattern;
    }

    @Override
    public boolean equals(Object other) {
      if (other instanceof ExactMatcher otherMatcher) {
        return this.rawPattern.equals(otherMatcher.rawPattern);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return this.rawPattern.hashCode();
    }
  }

  /** Checks if the given label fails to match the pattern. */
  private static final class NegativeMatcher implements SinglePatternMatcher {
    private final SinglePatternMatcher inner;

    private NegativeMatcher(SinglePatternMatcher inner) {
      this.inner = inner;
    }

    @Override
    public MatchResult matches(Label label) {
      return switch (this.inner.matches(label)) {
        case INCLUDE -> MatchResult.EXCLUDE;
        case EXCLUDE, NOT_RELEVANT -> MatchResult.NOT_RELEVANT;
      };
    }

    @Override
    public String toString() {
      return String.format("-%s", this.inner);
    }

    @Override
    public boolean equals(Object other) {
      if (other instanceof NegativeMatcher otherMatcher) {
        return this.inner.equals(otherMatcher.inner);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return 0x37 ^ this.inner.hashCode();
    }
  }

  private static final class WildcardMatcher implements SinglePatternMatcher {
    private final PathFragment packagePath;

    private WildcardMatcher(String pattern) {
      // Strip off the leading "//" and the trailing "/..." and create the wildcard matcher.
      this.packagePath = PathFragment.create(pattern.substring(2, pattern.lastIndexOf("...")));
    }

    @Override
    public MatchResult matches(Label label) {
      if (label.getPackageFragment().startsWith(this.packagePath)) {
        return MatchResult.INCLUDE;
      }
      return MatchResult.NOT_RELEVANT;
    }

    @Override
    public String toString() {
      return String.format("//%s/...", this.packagePath);
    }

    @Override
    public boolean equals(Object other) {
      if (other instanceof WildcardMatcher otherMatcher) {
        return this.packagePath.equals(otherMatcher.packagePath);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return this.packagePath.hashCode();
    }
  }
}
