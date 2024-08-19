// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the how the visibility system works with respect to symbolic macros. */
@RunWith(JUnit4.class)
public final class MacroVisibilityTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    setBuildLanguageOptions("--experimental_enable_first_class_macros");
  }

  @Test
  public void todo() {
    // To be populated in immediate follow-up.
  }

  /*
   * TODO: #19922 - Tests cases to add:
   *
   * ---- Basic functionality  ----
   *
   * - BUILD file can and cannot see public and private target of a macro (whose code is in separate
   *   package). Same for implicit outputs of those targets
   *
   * - Macro can and cannot see public and private target of the BUILD file it's instantiated in
   *   (different from its code's location). Same for implicit outputs of those targets.
   *
   * - Two sibling macros in the same BUILD file (two different code location packages) can see each
   *   other's public details and not each other's private details.
   *
   * - BUILD file instantiates outer macro which instantiates inner macro, each macro's code defined
   *   in its own package. Separate package defines targets visible to only inner macro and outer
   *   macro. Inner and outer macros can only see the one visible to it, not the other.
   *
   * - cc_library defined in helper func in //A, called from impl func in //B, used in macro() def
   *   in //C, loaded and exported by //D, called by legacy macro defined in //E, called in BUILD
   *   file in //F. Confirm that the cc_library can see a target of another separate package that
   *   gives visibility only to //D.
   *
   * ---- Visibility violations don't block more than they have to ----
   *
   * - Vis violation on target doesn't block building sibling target in same macro.
   * - Vis violation on macro doesn't block building targets of other macros.
   * - Vis violation on macro doesn't block building targets in same macro that don't use the
   *   forbidden prereq.
   *
   * ---- Propagating target usages from parent macro to child ----
   *
   * - An inner macro can see a target it doesn't have permission on, if the parent macro has
   *   permission and the parent passes the label into the inner macro as an attribute of the inner
   *   macro.
   *
   * - This doesn't work if the parent macro passes the label but doesn't itself have permission.
   *
   * - This doesn't work if the parent macro has permission but doesn't pass the label in. Implicit
   *   deps of the inner macro do not qualify as the outer macro passing it in.
   *
   * - It's an error if the parent passes the label in but does not have permission, even if the
   *   inner macro independently has its own permission.
   *
   * - Permission can be passed recursively through multiple levels, but not through a gap (e.g.
   *   middle macro does not have permission or does not thread it through).
   *
   * - But if there is a gap, and if the inner macro properly has permission and does not get passed
   *   the label from the middle, then the original usage by the outer macro (which is independent /
   *   inconsequential to the usage by the target in the inner macro) is not validated and could
   *   erroneously pass.
   *
   * ---- Implicit deps ----
   *
   * - If a rule's implicit dep is defined in a macro, the check of the rule's def loc against the
   *   dep takes into account the macro's location. In particular, a rule can see implicit deps
   *   defined by macros whose defs are in the same package as the rule's def (when the macro is
   *   called from a totally different package), even if the dep is otherwise private.
   *
   * - If a macro has an implicit dep, that dep's visibility is checked against the macro def's
   *   location, not its instantiation location. So pass even when the instance doesn't have
   *   permission. And fail if the macro def location doesn't have permission, even if the instance
   *   does.
   *
   * ---- Visibility attr usage ----
   *
   * - Visibility attr is passed and contains the call site's package.
   *
   * - Exporting via visibility = visibility works, including transitively.
   *
   * - Passing visibility to a macro does not force that visibility upon the macro's internal
   *   targets that don't declare a visibility.
   *
   * - Can compose public and private visibilities with other visibilities via concatenation.
   *
   * ---- default_visibility ----
   *
   * - default_visibility does not propagate to inside any symbolic macro, to either macros or
   *   targets.
   *
   * - default_visibility affects the visibility of a top-level macro that does not set
   *   visibility=..., and does not affect a top-level macro that does set visibility=...
   *
   * ---- Visibility attr representation ----
   *
   * - Visibility attr is normalized. (Might be blocked on changing legacy visibility system's
   *   representation for visibility attr of targets.)
   *
   * - default_visibility is inlined into targets' visibilities. (Again, might be blocked.)
   *
   * ---- Accounting for CommonPrerequisiteValidator#isSameLogicalPackage() ----
   *
   * - When appending the call site location to the given visibility of a declaration, also append
   *   other equivalent packages determined by isSameLogicalPackage().
   *
   * - Don't do this for ordinary entries specified explicitly in the visibility attr.
   */

  // TODO: #19922 - Consider any other edge cases regarding exotic dependencies and other
  // PrerequisiteValidator code paths, e.g. implicit deps, toolchain resolution, etc.

  // TODO: #19922 - Test that the new visibility system is compatible with the old on builds that do
  // not have symbolic macros. Then delete the old system in favor of the new.
}
