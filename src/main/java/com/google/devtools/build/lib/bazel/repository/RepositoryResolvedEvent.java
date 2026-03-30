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

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.collect.Maps;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Event indicating that a repository rule was executed, together with the return value of the rule.
 */
public class RepositoryResolvedEvent {

  private final boolean informationReturned;
  private final String message;

  public RepositoryResolvedEvent(RepoDefinition repoDefinition, Map<String, Object> result) {
    if (result.isEmpty()) {
      // Repo claims to be already reproducible, so wants to be called as is.
      this.informationReturned = false;
      this.message = "Repo '" + repoDefinition.name() + "' finished fetching.";
    } else {
      // Repo claims that the returned (probably changed) arguments are a reproducible
      // version of itself. Diff them and report the changes, if any.
      var modifiedAttributes =
          repoDefinition.getFieldNames().stream()
              // The "name" attribute is confusing as the value specified by the user is transformed
              // to the canonical name for repository_ctx.attr.name. Since the name should never
              // affect reproducibility, ignore it.
              .filter(name -> !name.equals("name"))
              // Filter out implicit attributes, which can't be modified by the user.
              .filter(name -> !name.startsWith("_"))
              .map(
                  name -> {
                    var defaultValue =
                        repoDefinition
                            .repoRule()
                            .attributes()
                            .get(repoDefinition.repoRule().attributeIndices().get(name))
                            .getDefaultValueUnchecked();
                    // Label attributes report a default of null rather than None.
                    if (defaultValue == null) {
                      defaultValue = Starlark.NONE;
                    }
                    var currentValue = repoDefinition.getValue(name);
                    var newValue = result.getOrDefault(name, defaultValue);
                    if (newValue.equals(currentValue)) {
                      return null;
                    }
                    // Distinguish between "dropped" and non-trivially "modified" attributes.
                    return Map.entry(
                        name,
                        newValue.equals(defaultValue) ? Optional.empty() : Optional.of(newValue));
                  })
              .filter(Objects::nonNull)
              .collect(toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));
      if (modifiedAttributes.isEmpty()) {
        this.informationReturned = false;
        this.message = "Repo '" + repoDefinition.name() + "' finished fetching.";
      } else {
        this.informationReturned = true;
        var modifiedToNonDefault = Maps.filterValues(modifiedAttributes, Optional::isPresent);
        var dropped = Maps.filterValues(modifiedAttributes, Optional::isEmpty).keySet();
        if (modifiedToNonDefault.isEmpty()) {
          this.message =
              "Repo '"
                  + repoDefinition.name()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " dropping arguments "
                  + Starlark.repr(dropped, StarlarkSemantics.DEFAULT);
        } else if (dropped.isEmpty()) {
          this.message =
              "Repo '"
                  + repoDefinition.name()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " modifying arguments "
                  + representModifications(
                      Maps.transformValues(modifiedToNonDefault, Optional::get));
        } else {
          this.message =
              "Repo '"
                  + repoDefinition.name()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " modifying arguments "
                  + representModifications(
                      Maps.transformValues(modifiedToNonDefault, Optional::get))
                  + " and dropping "
                  + Starlark.repr(dropped, StarlarkSemantics.DEFAULT);
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

  static String representModifications(Map<String, Object> changes) {
    return changes.entrySet().stream()
        .map(
            entry ->
                "%s = %s"
                    .formatted(
                        entry.getKey(), Starlark.repr(entry.getValue(), StarlarkSemantics.DEFAULT)))
        .collect(Collectors.joining(", "));
  }
}
