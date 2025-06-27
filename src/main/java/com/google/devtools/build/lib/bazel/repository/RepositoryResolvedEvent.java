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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.util.Pair;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/**
 * Event indicating that a repository rule was executed, together with the return value of the rule.
 */
public class RepositoryResolvedEvent {

  private final boolean informationReturned;
  private final String message;

  public RepositoryResolvedEvent(Rule rule, StructImpl attrs, Map<?, ?> result) {
    ImmutableMap.Builder<String, Object> origAttrBuilder = ImmutableMap.builder();
    ImmutableMap.Builder<String, Object> defaults = ImmutableMap.builder();

    for (Attribute attr : rule.getAttributes()) {
      if (!attr.isPublic()) {
        continue;
      }
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
    ImmutableMap<String, Object> origAttr = origAttrBuilder.buildOrThrow();

    if (result.isEmpty()) {
      // Rule claims to be already reproducible, so wants to be called as is.
      this.informationReturned = false;
      this.message = "Repository rule '" + rule.getName() + "' finished.";
    } else {
      // Rule claims that the returned (probably changed) arguments are a reproducible
      // version of itself.
      Pair<Map<String, Object>, List<String>> diff =
          compare(origAttr, defaults.buildOrThrow(), result);
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
                  + Starlark.repr(diff.getSecond());
        } else if (diff.getSecond().isEmpty()) {
          this.message =
              "Rule '"
                  + rule.getName()
                  + "' indicated that a canonical reproducible form can be obtained by"
                  + " modifying arguments "
                  + representModifications(diff.getFirst());
        } else {
          this.message =
              "Rule '"
                  + rule.getName()
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
  public static String getRuleDefinitionInformation(Rule rule) {
    StringBuilder buf = new StringBuilder();

    // Emit stack of rule instantiation.
    buf.append("Repository ").append(rule.getName()).append(" instantiated at:\n");
    ImmutableList<StarlarkThread.CallStackEntry> stack = rule.reconstructCallStack();
    // TODO: Callstack should always be available for bazel.
    if (stack.isEmpty()) {
      buf.append("  callstack not available\n");
    } else {
      for (StarlarkThread.CallStackEntry frame : stack) {
        buf.append("  ").append(frame.location).append(": in ").append(frame.name).append('\n');
      }
    }

    // Emit stack of rule class declaration.
    stack = rule.getRuleClassObject().getCallStack();
    if (stack.isEmpty()) {
      buf.append("Repository rule ").append(rule.getRuleClass()).append(" is built-in.\n");
    } else {
      buf.append("Repository rule ").append(rule.getRuleClass()).append(" defined at:\n");
      for (StarlarkThread.CallStackEntry frame : stack) {
        buf.append("  ").append(frame.location).append(": in ").append(frame.name).append('\n');
      }
    }

    return buf.toString();
  }

  /**
   * Attributes that may be defined on a repository rule without affecting its canonical
   * representation. These may be created implicitly by Bazel.
   */
  private static final ImmutableSet<String> IGNORED_ATTRIBUTE_NAMES =
      ImmutableSet.of("generator_name", "generator_function", "generator_location");

  /**
   * Compare two maps from Strings to objects, returning a pair of the map with all entries not in
   * the original map or in the original map, but with a different value, and the keys dropped from
   * the original map. However, ignore changes where a value is explicitly set to its default.
   *
   * <p>Ignores attributes listed in {@code IGNORED_ATTRIBUTE_NAMES}.
   */
  static Pair<Map<String, Object>, List<String>> compare(
      Map<String, Object> orig, Map<String, Object> defaults, Map<?, ?> modified) {
    ImmutableMap.Builder<String, Object> valuesChanged = ImmutableMap.builder();
    for (Map.Entry<?, ?> entry : modified.entrySet()) {
      if (entry.getKey() instanceof String key) {
        if (IGNORED_ATTRIBUTE_NAMES.contains(key)) {
          // The dict returned by the repo rule really shouldn't know about these anyway, but
          // for symmetry we'll ignore them if they happen to be present.
          continue;
        }
        Object value = entry.getValue();
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
    ImmutableList.Builder<String> keysDropped = ImmutableList.builder();
    for (String key : orig.keySet()) {
      if (IGNORED_ATTRIBUTE_NAMES.contains(key)) {
        continue;
      }
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
