// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.repository;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import java.util.Map;

/**
 * Event indicating that a repository rule was executed, together with the return value of the rule.
 */
public class RepositoryResolvedEvent implements Postable {
  public static final String ORIGINAL_RULE_CLASS = "original_rule_class";
  public static final String ORIGINAL_ATTRIBUTES = "original_attributes";
  public static final String RULE_CLASS = "rule_class";
  public static final String ATTRIBUTES = "attributes";
  public static final String REPOSITORIES = "repositories";

  /**
   * The entry for WORSPACE.resolved corresponding to that rule invocation.
   *
   * <p>It will always be a dict with three entries <ul>
   *  <li> the original rule class (as String, e.g., "@bazel_tools//:git.bzl%git_repository")
   *  <li> the original attributes (as dict, e.g., mapping "name" to "build_bazel"
   *       and "remote" to "https://github.com/bazelbuild/bazel.git"), and
   *  <li> a "repositories" entry; this is a list, often a single entry, of fully resolved
   *       repositories the rule call expanded to (in the above example, the attributes entry
   *       would have an additional "commit" and "shallow-since" entry).
   * </ul>
   */
  private final Object resolvedInformation;

  public RepositoryResolvedEvent(Rule rule, Info attrs, Object result) {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();

    String originalClass =
        rule.getRuleClassObject().getRuleDefinitionEnvironmentLabel() + "%" + rule.getRuleClass();
    builder.put(ORIGINAL_RULE_CLASS, originalClass);

    ImmutableMap.Builder<String, Object> origAttrBuilder = ImmutableMap.builder();
    for (Attribute attr : rule.getAttributes()) {
      String name = attr.getPublicName();
      if (!name.startsWith("_")) {
        // TODO(aehlig): filter out remaining attributes that cannot be set in a
        // WORKSPACE file.
        try {
          Object value = attrs.getValue(name, Object.class);
          // Only record explicit values, skip computed defaults
          if (!(value instanceof Attribute.ComputedDefault)) {
            origAttrBuilder.put(name, value);
          }
        } catch (EvalException e) {
          // Do nothing, just ignore the value.
        }
      }
    }
    ImmutableMap<String, Object> origAttr = origAttrBuilder.build();
    builder.put(ORIGINAL_ATTRIBUTES, origAttr);

    if (result == Runtime.NONE) {
      // Rule claims to be already reproducible, so wants to be called as is.
      builder.put(
          REPOSITORIES,
          ImmutableList.<Object>of(
              ImmutableMap.<String, Object>builder()
                  .put(RULE_CLASS, originalClass)
                  .put(ATTRIBUTES, origAttr)
                  .build()));
    } else if (result instanceof Map) {
      // Rule claims that the returned (probably changed) arguments are a reproducible
      // version of itself.
      builder.put(
          REPOSITORIES,
          ImmutableList.<Object>of(
              ImmutableMap.<String, Object>builder()
                  .put(RULE_CLASS, originalClass)
                  .put(ATTRIBUTES, result)
                  .build()));
    } else {
      // TODO(aehlig): handle strings specially to allow encodings of the former
      // values to be accepted as well.
      builder.put(REPOSITORIES, result);
    }

    this.resolvedInformation = builder.build();
  }

  /** Return the entry for the given rule invocation in a format suitable for WORKSPACE.resolved. */
  public Object getResolvedInformation() {
    return resolvedInformation;
  }
}
