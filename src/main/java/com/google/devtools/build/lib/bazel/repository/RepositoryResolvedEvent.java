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

import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.ATTRIBUTES;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.ORIGINAL_ATTRIBUTES;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.ORIGINAL_RULE_CLASS;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.OUTPUT_TREE_HASH;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.REPOSITORIES;
import static com.google.devtools.build.lib.rules.repository.ResolvedHashesFunction.RULE_CLASS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Map;

/**
 * Event indicating that a repository rule was executed, together with the return value of the rule.
 */
public class RepositoryResolvedEvent implements ProgressLike {

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

  private final String name;

  public RepositoryResolvedEvent(Rule rule, StructImpl attrs, Path outputDirectory, Object result) {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();

    String originalClass =
        rule.getRuleClassObject().getRuleDefinitionEnvironmentLabel() + "%" + rule.getRuleClass();
    builder.put(ORIGINAL_RULE_CLASS, originalClass);

    ImmutableMap.Builder<String, Object> origAttrBuilder = ImmutableMap.builder();
    for (Attribute attr : rule.getAttributes()) {
      if (rule.isAttributeValueExplicitlySpecified(attr)) {
        String name = attr.getPublicName();
        try {
          Object value = attrs.getValue(name, Object.class);
          origAttrBuilder.put(name, value);
        } catch (EvalException e) {
          // Do nothing, just ignore the value.
        }
      }
    }
    ImmutableMap<String, Object> origAttr = origAttrBuilder.build();
    builder.put(ORIGINAL_ATTRIBUTES, origAttr);

    ImmutableMap.Builder<String, Object> repositoryBuilder =
        ImmutableMap.<String, Object>builder().put(RULE_CLASS, originalClass);

    try {
      repositoryBuilder.put(OUTPUT_TREE_HASH, outputDirectory.getDirectoryDigest());
    } catch (IOException e) {
      // Digest not available, but we still have to report that a repository rule
      // was invoked. So we can do nothing, but ignore the event.
    }

    if (result == Runtime.NONE) {
      // Rule claims to be already reproducible, so wants to be called as is.
      builder.put(
          REPOSITORIES,
          ImmutableList.<Object>of(repositoryBuilder.put(ATTRIBUTES, origAttr).build()));
    } else if (result instanceof Map) {
      // Rule claims that the returned (probably changed) arguments are a reproducible
      // version of itself.
      builder.put(
          REPOSITORIES,
          ImmutableList.<Object>of(repositoryBuilder.put(ATTRIBUTES, result).build()));
    } else {
      // TODO(aehlig): handle strings specially to allow encodings of the former
      // values to be accepted as well.
      builder.put(REPOSITORIES, result);
    }

    this.resolvedInformation = builder.build();
    this.name = rule.getName();
  }

  /** Return the entry for the given rule invocation in a format suitable for WORKSPACE.resolved. */
  public Object getResolvedInformation() {
    return resolvedInformation;
  }

  /** Return the name of the rule that produced the resolvedInformation */
  public String getName() {
    return name;
  }
}
