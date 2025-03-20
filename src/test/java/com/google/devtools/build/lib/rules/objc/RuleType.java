// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.Map;
import java.util.Set;

/**
 * Provides utilities to help test a certain rule type without requiring the calling code to know
 * exactly what kind of rule is being tested. Only one instance is needed per rule type (e.g. one
 * instance for {@code objc_library}).
 */
public abstract class RuleType {
  /**
   * What to pass as the value of some attribute to indicate an attribute should not be added to the
   * rule. This can either be to test an error condition, or to use an alternative attribute to
   * supply the value.
   */
  public static final String OMIT_REQUIRED_ATTR = "<OMIT_REQUIRED_ATTR>";

  private final String ruleTypeName;

  RuleType(String ruleTypeName) {
    this.ruleTypeName = ruleTypeName;
  }

  /**
   * The name of this type as it appears in {@code BUILD} files, such as {@code objc_library}.
   */
  final String getRuleTypeName() {
    return ruleTypeName;
  }

  /**
   * Returns names and values, and otherwise prepares, extra attributes required for this rule type
   * to be without error. For instance, if this rule type requires 'srcs' and 'infoplist'
   * attributes, this method may be implemented as follows:
   * <pre>
   * {@code
   * List<String> attributes = new ArrayList<>();
   * if (!alreadyAdded.contains("srcs")) {
   *   scratch.file("/workspace_root/" + packageDir + "/a.m");
   *   attributes.add("srcs = ['a.m']");
   * }
   * if (!alreadyAdded.contains(INFOPLIST_ATTR)) {
   *   scratch.file("/workspace_root/" + packageDir + "Info.plist");
   *   attributes.add("infoplist = ['Info.plist']");
   * }
   * return attributes;
   * </pre>
   * }
   *
   * @throws IOException for whatever reason the implementer feels like, but mostly just when
   *     a scratch file couldn't be created
   */
  abstract Iterable<String> requiredAttributes(
      Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException;

  private ImmutableMap<String, String> map(String... attrs) {
    ImmutableMap.Builder<String, String> map = new ImmutableMap.Builder<>();
    Preconditions.checkArgument((attrs.length & 1) == 0,
        "attrs must have an even number of elements");
    for (int i = 0; i < attrs.length; i += 2) {
      map.put(attrs[i], attrs[i + 1]);
    }
    return map.build();
  }

  /**
   * Generates the String necessary to define a target of this rule type.
   *
   * @param packageDir the package in which to create the target
   * @param name the name of the target
   * @param checkSpecificAttrs alternating name/values of attributes to add to the rule that are
   *     required for the check being performed to be defined a certain way. Pass
   *     {@link #OMIT_REQUIRED_ATTR} for a value to prevent an attribute from being automatically
   *     defined.
   */
  final String target(
      Scratch scratch, String packageDir, String name, String... checkSpecificAttrs)
      throws IOException {
    ImmutableMap<String, String> checkSpecific = map(checkSpecificAttrs);
    StringBuilder target = new StringBuilder(ruleTypeName)
        .append("(name = '")
        .append(name)
        .append("',");
    for (Map.Entry<String, String> entry : checkSpecific.entrySet()) {
      if (entry.getValue().equals(OMIT_REQUIRED_ATTR)) {
        continue;
      }
      target.append(entry.getKey())
          .append("=")
          .append(entry.getValue())
          .append(",");
    }
    Joiner.on(",").appendTo(
        target,
        requiredAttributes(scratch, packageDir, checkSpecific.keySet()));
    target.append(')');
    return target.toString();

  }

  /**
   * Creates a target at //x:x which is the only target in the BUILD file. Returns the string that
   * is written to the scratch file as it is often useful for debugging purposes.
   */
  public final String scratchTarget(Scratch scratch, String... checkSpecificAttrs)
      throws IOException {
    return scratchTarget("x", "x", scratch, checkSpecificAttrs);
  }

  /**
   * Creates a target at a given package which is the only target in the BUILD file. Returns the
   * string that is written to the scratch file as it is often useful for debugging purposes.
   *
   * @param packageDir the package of the target, for example "foo" in //foo:bar
   * @param targetName the name of the target, for example "bar" in //foo:bar
   * @param scratch the scratch object to use to create the build file
   * @param checkSpecificAttrs alternating name/values of attributes to add to the rule that are
   *     required for the check being performed to be defined a certain way. Pass
   *     {@link #OMIT_REQUIRED_ATTR} for a value to prevent an attribute from being automatically
   *     defined.
   */
  public final String scratchTarget(String packageDir, String targetName,
      Scratch scratch, String... checkSpecificAttrs)
      throws IOException {
    String target = target(scratch, packageDir, targetName, checkSpecificAttrs);
    scratch.file(packageDir + "/BUILD", starlarkLoadPrerequisites() + "\n" + target);
    return target;
  }

  /**
   * Returns a string (of one or more lines) required by BUILD files which reference targets of this
   * rule type.
   *
   * <p>Subclasses of {@link RuleType} should override this method if using the rule requires
   * Starlark files to be loaded.
   */
  public String starlarkLoadPrerequisites() {
    return "";
  }
}
