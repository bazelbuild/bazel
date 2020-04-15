// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.constraints;

import com.google.devtools.build.lib.analysis.constraints.SupportedEnvironmentsProvider.RemovedEnvironmentCulprit;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Implementation of the semantics of Bazel's constraint specification and enforcement system.
 *
 * <p>This is how the system works:
 *
 * <p>All build rules can declare which "static environments" they can be built for, where a "static
 * environment" is a label instance of an {@link EnvironmentRule} rule declared in a BUILD file.
 * There are various ways to do this:
 *
 * <ul>
 *   <li>Through a "restricted to" attribute setting ({@link
 *       com.google.devtools.build.lib.packages.RuleClass#RESTRICTED_ENVIRONMENT_ATTR}). This is the
 *       most direct form of specification - it declares the exact set of environments the rule
 *       supports (for its group - see precise details below).
 *   <li>Through a "compatible with" attribute setting ({@link
 *       com.google.devtools.build.lib.packages.RuleClass#COMPATIBLE_ENVIRONMENT_ATTR}. This
 *       declares <b>additional</b> environments a rule supports in addition to "standard"
 *       environments that are supported by default (see below).
 *   <li>Through "default" specifications in {@link EnvironmentGroup} rules. Every environment
 *       belongs to a group of thematically related peers (e.g. "target architectures", "JDK
 *       versions", or "mobile devices"). An environment group's definition includes which of these
 *       environments should be supported "by default" if not otherwise specified by one of the
 *       above mechanisms. In particular, a rule with no environment-related attributes
 *       automatically inherits all defaults.
 *   <li>Through a rule class default ({@link
 *       com.google.devtools.build.lib.packages.RuleClass.Builder#restrictedTo} and {@link
 *       com.google.devtools.build.lib.packages.RuleClass.Builder#compatibleWith}). This overrides
 *       global defaults for all instances of the given rule class. This can be used, for example,
 *       to make all *_test rules "testable" without each instance having to explicitly declare this
 *       capability.
 * </ul>
 *
 * <p>Groups exist to model the idea that some environments are related while others have nothing to
 * do with each other. Say, for example, we want to say a rule works for PowerPC platforms but not
 * x86. We can do so by setting its "restricted to" attribute to {@code ['//sample/path:powerpc']}.
 * Because both PowerPC and x86 are in the same "target architectures" group, this setting removes
 * x86 from the set of supported environments. But since JDK support belongs to its own group ("JDK
 * versions") it says nothing about which JDK the rule supports.
 *
 * <p>More precisely, if a rule has a "restricted to" value of [A, B, C], this removes support for
 * all default environments D such that group(D) is in [group(A), group(B), group(C)] AND D is not
 * in [A, B, C] (in other words, D isn't explicitly opted back in). The rule's full set of supported
 * environments thus becomes [A, B, C] + all defaults that belong to unrelated groups.
 *
 * <p>If the rule has a "compatible with" value of [E, F, G], these are unconditionally added to its
 * set of supported environments (in addition to the results from above).
 *
 * <p>An environment may not appear in both a rule's "restricted to" and "compatible with" values.
 * If two environments belong to the same group, they must either both be in "restricted to", both
 * be in "compatible with", or not explicitly specified.
 *
 * <p>Given all the above, constraint enforcement is this: rule A can depend on rule B if, for every
 * static environment A supports, B also supports that environment.
 *
 * <p>Configurable attributes introduce the additional concept of "refined environments". Given:
 *
 * <pre>
 *   java_library(
 *       name = "lib",
 *       restricted_to = [":A", ":B"],
 *       deps = select({
 *           ":config_a": [":depA"],
 *           ":config_b": [":depB"],
 *       }))
 *   java_library(
 *       name = "depA",
 *       restricted_to = [":A"])
 *   java_library(
 *       name = "depB",
 *       restricted_to = [":B"])
 * </pre>
 *
 * "lib"'s static environments are what are declared via restricted_to: {@code [":A", ":B"]}. But
 * normal constraint checking doesn't work well here: neither "depA" or "depB" supports both
 * environments, so each is technically invalid. But the two of them together <i>do</i> support both
 * environments. So constraint checking with selects checks that "lib"'s environments are supported
 * by the <i>union</i> of its selectable dependencies, then <i>refines</i> its environments to
 * whichever deps get chosen. In other words:
 *
 * <ol>
 *   <li>The above example is considered constraint-valid.
 *   <li>When building with "config_a", "lib"'s refined environment set is {@code [":A"]}.
 *   <li>When building with "config_b", "lib"'s refined environment set is {@code [":B"]}.
 *   <li>Any rule depending on "lib" has its environments refined by the intersection with "lib". So
 *       if "depender" has {@code restricted_to = [":A", ":B"]} and {@code deps = [":lib"]}, then
 *       when building with "config_a", "depender"'s refined environment set is {@code [":A"]}.
 *   <li>For each environment group, every rule's refined environment set must be non-empty. This
 *       ensures the "chosen" dep in a select matches all rules up the dependency chain. So if
 *       "depender" had {@code restricted_to = [":B"]}, it wouldn't be allowed in a "config_a"
 *       build.
 * </ol>
 *
 * </code>.
 *
 * @param <T> The type of object to check for constraints.
 */
public interface ConstraintSemantics<T> {

  /**
   * Returns the environment group that owns the given environment. Both must belong to the same
   * package.
   *
   * @throws EnvironmentLookupException if the input is not an {@link EnvironmentRule} or no
   *     matching group is found
   */
  static EnvironmentGroup getEnvironmentGroup(Target envTarget) throws EnvironmentLookupException {
    if (!(envTarget instanceof Rule)
        || !((Rule) envTarget).getRuleClass().equals(ConstraintConstants.ENVIRONMENT_RULE)) {
      throw new EnvironmentLookupException(
          envTarget.getLabel() + " is not a valid environment definition");
    }
    for (EnvironmentGroup group : envTarget.getPackage().getTargets(EnvironmentGroup.class)) {
      if (group.getEnvironments().contains(envTarget.getLabel())) {
        return group;
      }
    }
    throw new EnvironmentLookupException(
        "cannot find the group for environment " + envTarget.getLabel());
  }

  /**
   * Returns the set of environments this rule supports.
   *
   * <p>Note this set is <b>not complete</b> - it doesn't include environments from groups we don't
   * "know about". Environments and groups can be declared in any package. If the rule includes no
   * references to that package, then it simply doesn't know anything about them. But the constraint
   * semantics say the rule should support the defaults for that group. We encode this implicitly:
   * given the returned set, for any group that's not in the set the rule is also considered to
   * support that group's defaults.
   *
   * @param context analysis context for the rule. A rule error is triggered here if invalid
   *     constraint settings are discovered.
   * @return the environments this rule supports, not counting defaults "unknown" to this rule as
   *     described above. Returns null if any errors are encountered.
   */
  @Nullable
  EnvironmentCollection getSupportedEnvironments(T context);

  /**
   * Performs constraint checking on the given rule's dependencies and reports any errors. This
   * includes:
   *
   * <ul>
   *   <li>Static environment checking: if this rule supports environment E, all deps outside
   *       selects must also support E
   *   <li>Refined environment computation: this rule's refined environments are its static
   *       environments intersected with the refined environments of all dependencies (including
   *       chosen deps in selects)
   *   <li>Refined environment checking: no environment groups can be "emptied" due to refinement
   * </ul>
   *
   * @param context the rule to analyze
   * @param staticEnvironments the rule's supported environments, as defined by the return value of
   *     {@link #getSupportedEnvironments}. In particular, for any environment group that's not in
   *     this collection, the rule is assumed to support the defaults for that group.
   * @param refinedEnvironments a builder for populating this rule's refined environments
   * @param removedEnvironmentCulprits a builder for populating the core dependencies that trigger
   *     pruning away environments through refinement. If multiple dependencies qualify (e.g. two
   *     direct deps under the current rule), one is arbitrarily chosen.
   */
  void checkConstraints(
      T context,
      EnvironmentCollection staticEnvironments,
      EnvironmentCollection.Builder refinedEnvironments,
      Map<Label, RemovedEnvironmentCulprit> removedEnvironmentCulprits);

  /** Exception indicating errors finding/parsing environments or their containing groups. */
  class EnvironmentLookupException extends Exception {
    private EnvironmentLookupException(String message) {
      super(message);
    }
  }
}
