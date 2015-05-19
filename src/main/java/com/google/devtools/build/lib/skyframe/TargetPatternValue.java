// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.ResolvedTargets.Builder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;

/**
 * A value referring to a computed set of resolved targets. This is used for the results of target
 * pattern parsing.
 */
@Immutable
@ThreadSafe
public final class TargetPatternValue implements SkyValue {

  private ResolvedTargets<Label> targets;

  TargetPatternValue(ResolvedTargets<Label> targets) {
    this.targets = Preconditions.checkNotNull(targets);
  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    List<String> ts = new ArrayList<>();
    List<String> filteredTs = new ArrayList<>();
    for (Label target : targets.getTargets()) {
      ts.add(target.toString());
    }
    for (Label target : targets.getFilteredTargets()) {
      filteredTs.add(target.toString());
    }

    out.writeObject(ts);
    out.writeObject(filteredTs);
  }

  private Label labelFromString(String labelString) {
    try {
      return Label.parseAbsolute(labelString);
    } catch (SyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  @SuppressWarnings("unchecked")
  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    List<String> ts = (List<String>) in.readObject();
    List<String> filteredTs = (List<String>) in.readObject();

    Builder<Label> builder = ResolvedTargets.<Label>builder();
    for (String labelString : ts) {
      builder.add(labelFromString(labelString));
    }

    for (String labelString : filteredTs) {
      builder.remove(labelFromString(labelString));
    }
    this.targets = builder.build();
  }

  @SuppressWarnings("unused")
  private void readObjectNoData() {
    throw new IllegalStateException();
  }

  /**
   * Create a target pattern value key.
   *
   * @param pattern The pattern, eg "-foo/biz...". If the first character is "-", the pattern
   *                is treated as a negative pattern.
   * @param policy The filtering policy, eg "only return test targets"
   * @param offset The offset to apply to relative target patterns.
   */
  @ThreadSafe
  public static SkyKey key(String pattern,
                            FilteringPolicy policy,
                            String offset) {
    return new SkyKey(SkyFunctions.TARGET_PATTERN,
        pattern.startsWith("-")
        // Don't apply filters to negative patterns.
        ? new TargetPatternKey(pattern.substring(1), FilteringPolicies.NO_FILTER, true, offset)
        : new TargetPatternKey(pattern, policy, false, offset));
  }

  /**
   * Like above, but accepts a collection of target patterns for the same filtering policy.
   *
   * @param patterns The collection of patterns, eg "-foo/biz...". If the first character is "-",
   *                 the pattern is treated as a negative pattern.
   * @param policy The filtering policy, eg "only return test targets"
   * @param offset The offset to apply to relative target patterns.
   */
  @ThreadSafe
  public static Iterable<SkyKey> keys(Collection<String> patterns,
                                       FilteringPolicy policy,
                                       String offset) {
    List<SkyKey> keys = Lists.newArrayListWithCapacity(patterns.size());
    for (String pattern : patterns) {
      keys.add(key(pattern, policy, offset));
    }
     return keys;
   }

  public ResolvedTargets<Label> getTargets() {
    return targets;
  }

  /**
   * A TargetPatternKey is a tuple of pattern (eg, "foo/..."), filtering policy, a relative pattern
   * offset, and whether it is a positive or negative match.
   */
  @ThreadSafe
  public static class TargetPatternKey implements Serializable {
    private final String pattern;
    private final FilteringPolicy policy;
    private final boolean isNegative;

    private final String offset;

    public TargetPatternKey(String pattern, FilteringPolicy policy,
        boolean isNegative, String offset) {
      this.pattern = Preconditions.checkNotNull(pattern);
      this.policy = Preconditions.checkNotNull(policy);
      this.isNegative = isNegative;
      this.offset = offset;
    }

    public String getPattern() {
      return pattern;
    }

    public boolean isNegative() {
      return isNegative;
    }

    public FilteringPolicy getPolicy() {
      return policy;
    }

    public String getOffset() {
      return offset;
    }

    @Override
    public String toString() {
      return (isNegative ? "-" : "") + pattern;
    }

    @Override
    public int hashCode() {
      return Objects.hash(pattern, isNegative, policy, offset);
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof TargetPatternKey)) {
        return false;
      }
      TargetPatternKey other = (TargetPatternKey) obj;

      return other.isNegative == this.isNegative && other.pattern.equals(this.pattern) &&
          other.offset.equals(this.offset) && other.policy.equals(this.policy);
    }
  }
}
