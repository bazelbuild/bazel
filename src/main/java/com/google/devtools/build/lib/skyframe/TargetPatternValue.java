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
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A value referring to a computed set of resolved targets. This is used for the results of target
 * pattern parsing.
 */
@Immutable
@ThreadSafe
public final class TargetPatternValue implements SkyValue {

  private ResolvedTargets<Target> targets;

  TargetPatternValue(ResolvedTargets<Target> targets) {
    this.targets = Preconditions.checkNotNull(targets);
  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    Set<Package> packages = new LinkedHashSet<>();
    List<String> ts = new ArrayList<>();
    List<String> filteredTs = new ArrayList<>();
    for (Target target : targets.getTargets()) {
      packages.add(target.getPackage());
      ts.add(target.getLabel().toString());
    }
    for (Target target : targets.getFilteredTargets()) {
      packages.add(target.getPackage());
      filteredTs.add(target.getLabel().toString());
    }

    out.writeObject(packages);
    out.writeObject(ts);
    out.writeObject(filteredTs);
  }

  @SuppressWarnings("unchecked")
  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    Set<Package> packages = (Set<Package>) in.readObject();
    List<String> ts = (List<String>) in.readObject();
    List<String> filteredTs = (List<String>) in.readObject();

    Map<String, Package> packageMap = new HashMap<>();
    for (Package p : packages) {
      packageMap.put(p.getName(), p);
    }

    Builder<Target> builder = ResolvedTargets.<Target>builder();
    for (String labelString : ts) {
      builder.add(lookupTarget(packageMap, labelString));
    }

    for (String labelString : filteredTs) {
      builder.remove(lookupTarget(packageMap, labelString));
    }
    this.targets = builder.build();
  }

  private static Target lookupTarget(Map<String, Package> packageMap, String labelString) {
    Label label = Label.parseAbsoluteUnchecked(labelString);
    Package p = packageMap.get(label.getPackageName());
    try {
      return p.getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      throw new IllegalStateException(e);
    }
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
        ? new TargetPattern(pattern.substring(1), FilteringPolicies.NO_FILTER, true, offset)
        : new TargetPattern(pattern, policy, false, offset));
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

  public ResolvedTargets<Target> getTargets() {
    return targets;
  }

  /**
   * A TargetPattern is a tuple of pattern (eg, "foo/..."), filtering policy, a relative pattern
   * offset, and whether it is a positive or negative match.
   */
  @ThreadSafe
  public static class TargetPattern implements Serializable {
    private final String pattern;
    private final FilteringPolicy policy;
    private final boolean isNegative;

    private final String offset;

    public TargetPattern(String pattern, FilteringPolicy policy,
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
      if (!(obj instanceof TargetPattern)) {
        return false;
      }
      TargetPattern other = (TargetPattern) obj;

      return other.isNegative == this.isNegative && other.pattern.equals(this.pattern) &&
          other.offset.equals(this.offset) && other.policy.equals(this.policy);
    }
  }
}
