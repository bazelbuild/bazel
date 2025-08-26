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
import com.google.devtools.build.lib.util.Pair;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Starlark;

/**
 * Event indicating that a repository rule was executed, together with the return value of the rule.
 */
public class RepositoryResolvedEvent {

  private final boolean informationReturned;
  private final String message;

  public RepositoryResolvedEvent(RepoDefinition repoDefinition, Map<?, ?> result) {
    if (result.isEmpty()) {
      // Repo claims to be already reproducible, so wants to be called as is.
      this.informationReturned = false;
      this.message = "Repo '" + repoDefinition.name() + "' finished fetching.";
    } else {
      // Repo claims that the returned (probably changed) arguments are a reproducible
      // version of itself.
      Pair<Map<String, Object>, List<String>> diff =
          compare(repoDefinition.attrValues().attributes(), result);
      if (diff.getFirst().isEmpty() && diff.getSecond().isEmpty()) {
        this.informationReturned = false;
        this.message = "Repo '" + repoDefinition.name() + "' finished fetching.";
      } else {
        this.informationReturned = true;
        if (diff.getFirst().isEmpty()) {
          this.message =
              "Repo '"
                  + repoDefinition.name()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " dropping arguments "
                  + Starlark.repr(diff.getSecond());
        } else if (diff.getSecond().isEmpty()) {
          this.message =
              "Repo '"
                  + repoDefinition.name()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " modifying arguments "
                  + representModifications(diff.getFirst());
        } else {
          this.message =
              "Repo '"
                  + repoDefinition.name()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " modifying arguments "
                  + representModifications(diff.getFirst())
                  + " and dropping "
                  + Starlark.repr(diff.getSecond());
        }
      }
    }
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

  /** Returns an unstructured message explaining the origin of this rule. */
  public static String getRuleDefinitionInformation(RepoDefinition repoDefinition) {
    // We used to output a call stack for repos defined in WORKSPACE, but in Bzlmod we always get an
    // empty stack -- for repos backing modules there's no call stack; for extension-generated
    // repos, the call stack is lost during the roundtrip to the lockfile.
    // TODO: store the call stack in the lockfile and output it here?
    return "Repo %s defined by rule %s in %s"
        .formatted(
            repoDefinition.name(),
            repoDefinition.repoRule().id().ruleName(),
            repoDefinition.repoRule().id().bzlFileLabel().getUnambiguousCanonicalForm());
  }

  /**
   * Compare two maps from Strings to objects, returning a pair of the map with all entries not in
   * the original map or in the original map, but with a different value, and the keys dropped from
   * the original map. However, ignore changes where a value is explicitly set to its default.
   */
  static Pair<Map<String, Object>, List<String>> compare(
      Map<String, Object> orig, Map<?, ?> modified) {
    ImmutableMap.Builder<String, Object> valuesChanged = ImmutableMap.builder();
    for (Map.Entry<?, ?> entry : modified.entrySet()) {
      if (entry.getKey() instanceof String key) {
        Object value = entry.getValue();
        if (!value.equals(orig.get(key))) {
          valuesChanged.put(key, value);
        }
      }
    }
    ImmutableList.Builder<String> keysDropped = ImmutableList.builder();
    for (String key : orig.keySet()) {
      if (!modified.containsKey(key)) {
        keysDropped.add(key);
      }
    }
    return Pair.of(valuesChanged.buildOrThrow(), keysDropped.build());
  }

  static String representModifications(Map<String, Object> changes) {
    StringBuilder representation = new StringBuilder();
    boolean isFirst = true;
    for (Map.Entry<String, Object> entry : changes.entrySet()) {
      if (!isFirst) {
        representation.append(", ");
      }
      representation.append(entry.getKey()).append(" = ").append(Starlark.repr(entry.getValue()));
      isFirst = false;
    }
    return representation.toString();
  }
}
