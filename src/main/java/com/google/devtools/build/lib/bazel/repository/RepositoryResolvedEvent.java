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
import com.google.devtools.build.lib.events.ExtendedEventHandler.ResolvedEvent;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Event indicating that a repository rule was executed, together with the return value of the rule.
 */
public class RepositoryResolvedEvent implements ResolvedEvent {

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
  private final boolean informationReturned;
  private final String message;
  private final String directoryDigest;

  public RepositoryResolvedEvent(Rule rule, StructImpl attrs, Path outputDirectory, Object result) {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();

    String originalClass =
        rule.getRuleClassObject().getRuleDefinitionEnvironmentLabel() + "%" + rule.getRuleClass();
    builder.put(ORIGINAL_RULE_CLASS, originalClass);

    ImmutableMap.Builder<String, Object> origAttrBuilder = ImmutableMap.builder();
    ImmutableMap.Builder<String, Object> defaults = ImmutableMap.builder();
    for (Attribute attr : rule.getAttributes()) {
      String name = attr.getPublicName();
      try {
        Object value = attrs.getValue(name, Object.class);
        if (value != null) {
          if (rule.isAttributeValueExplicitlySpecified(attr)) {
            origAttrBuilder.put(name, value);
          } else {
            defaults.put(name, value);
          }
        }
      } catch (EvalException e) {
        // Do nothing, just ignore the value.
      }
    }
    ImmutableMap<String, Object> origAttr = origAttrBuilder.build();
    builder.put(ORIGINAL_ATTRIBUTES, origAttr);

    ImmutableMap.Builder<String, Object> repositoryBuilder =
        ImmutableMap.<String, Object>builder().put(RULE_CLASS, originalClass);

    String digest = "[unavailable]";
    try {
      digest = outputDirectory.getDirectoryDigest();
      repositoryBuilder.put(OUTPUT_TREE_HASH, digest);
    } catch (IOException e) {
      // Digest not available, but we still have to report that a repository rule
      // was invoked. So we can do nothing, but ignore the event.
    }
    this.directoryDigest = digest;

    if (result == Runtime.NONE) {
      // Rule claims to be already reproducible, so wants to be called as is.
      builder.put(
          REPOSITORIES,
          ImmutableList.<Object>of(repositoryBuilder.put(ATTRIBUTES, origAttr).build()));
      this.informationReturned = false;
      this.message = "Repository rule '" + rule.getName() + "' finished.";
    } else if (result instanceof Map) {
      // Rule claims that the returned (probably changed) arguments are a reproducible
      // version of itself.
      builder.put(
          REPOSITORIES,
          ImmutableList.<Object>of(repositoryBuilder.put(ATTRIBUTES, result).build()));
      Pair<Map<String, Object>, List<String>> diff =
          compare(origAttr, defaults.build(), (Map<?, ?>) result);
      if (diff.getFirst().isEmpty() && diff.getSecond().isEmpty()) {
        this.informationReturned = false;
        this.message = "Repository rule '" + rule.getName() + "' finished.";
      } else {
        this.informationReturned = true;
        if (diff.getFirst().isEmpty()) {
          this.message =
              "Rule '"
                  + rule.getName()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " dropping arguments "
                  + Printer.getPrinter().repr(diff.getSecond());
        } else if (diff.getSecond().isEmpty()) {
          this.message =
              "Rule '"
                  + rule.getName()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " modifying arguments "
                  + Printer.getPrinter().repr(diff.getFirst());
        } else {
          this.message =
              "Rule '"
                  + rule.getName()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " modifying arguments "
                  + Printer.getPrinter().repr(diff.getFirst())
                  + " and dropping "
                  + Printer.getPrinter().repr(diff.getSecond());
        }
      }
    } else {
      // TODO(aehlig): handle strings specially to allow encodings of the former
      // values to be accepted as well.
      builder.put(REPOSITORIES, result);
      this.informationReturned = true;
      this.message = "Repository rule '" + rule.getName() + "' returned: " + result;
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

  public String getDirectoryDigest() {
    return directoryDigest;
  }

  /**
   * True, if the return value of the repository rule contained new information with respect to the
   * way it was called.
   */
  public boolean isNewInformationReturned() {
    return informationReturned;
  }

  /** Message describing the event */
  public String getMessage() {
    return message;
  }

  /**
   * Compare two maps from Strings to objects, returning a pair of the map with all entries not in
   * the original map or in the original map, but with a different value, and the keys dropped from
   * the original map. However, ignore changes where a value is explicitly set to its default.
   */
  static Pair<Map<String, Object>, List<String>> compare(
      Map<String, Object> orig, Map<String, Object> defaults, Map<?, ?> modified) {
    ImmutableMap.Builder<String, Object> valuesChanged = ImmutableMap.<String, Object>builder();
    for (Map.Entry<?, ?> entry : modified.entrySet()) {
      if (entry.getKey() instanceof String) {
        Object value = entry.getValue();
        String key = (String) entry.getKey();
        Object old = orig.get(key);
        if (old == null) {
          Object defaultValue = defaults.get(key);
          if (defaultValue == null || !defaultValue.equals(value)) {
            valuesChanged.put(key, value);
          }
        } else {
          if (!old.equals(entry.getValue())) {
            valuesChanged.put(key, value);
          }
        }
      }
    }
    ImmutableList.Builder<String> keysDropped = ImmutableList.<String>builder();
    for (String key : orig.keySet()) {
      if (!modified.containsKey(key)) {
        keysDropped.add(key);
      }
    }
    return Pair.of(valuesChanged.build(), keysDropped.build());
  }
}
