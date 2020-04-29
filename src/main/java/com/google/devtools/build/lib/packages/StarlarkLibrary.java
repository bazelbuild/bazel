// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.devtools.build.lib.packages.PackageFactory.getContext;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.util.List;
import java.util.Set;

/**
 * A library of pre-declared Bazel Starlark functions.
 *
 * <p>For functions pre-declared in a BUILD file, use {@link #BUILD}. For Bazel functions such as
 * {@code select} and {@code depset} that are pre-declared in all BUILD, .bzl, and WORKSPACE files,
 * use {@link #COMMON}. For functions pre-declared in every Starlark file, use {@link
 * Starlark#UNIVERSE}.
 */
public final class StarlarkLibrary {

  private StarlarkLibrary() {} // uninstantiable

  /**
   * A library of Starlark functions (keyed by name) that are not part of core Starlark but are
   * common to all Bazel Starlark file environments (BUILD, .bzl, and WORKSPACE). Examples: depset,
   * select.
   */
  public static final ImmutableMap<String, Object> COMMON = initCommon();

  private static ImmutableMap<String, Object> initCommon() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, new CommonLibrary());
    return env.build();
  }

  @SkylarkGlobalLibrary
  private static class CommonLibrary {

    @SkylarkCallable(
        name = "depset",
        doc =
            "Creates a <a href=\"depset.html\">depset</a>. The <code>direct</code> parameter is a"
                + " list of direct elements of the depset, and <code>transitive</code> parameter"
                + " is a list of depsets whose elements become indirect elements of the created"
                + " depset. The order in which elements are returned when the depset is converted"
                + " to a list is specified by the <code>order</code> parameter. See the <a"
                + " href=\"../depsets.md\">Depsets overview</a> for more information.\n" //
                + "<p>All"
                + " elements (direct and indirect) of a depset must be of the same type, as"
                + " obtained by the expression <code>type(x)</code>.\n" //
                + "<p>Because a hash-based set is used to eliminate duplicates during iteration,"
                + " all elements of a depset should be hashable. However, this invariant is not"
                + " currently checked consistently in all constructors. Use the"
                + " --incompatible_always_check_depset_elements flag to enable consistent"
                + " checking; this will be the default behavior in future releases;  see <a"
                + " href='https://github.com/bazelbuild/bazel/issues/10313'>Issue 10313</a>.\n" //
                + "<p>In addition, elements must currently be immutable, though this restriction"
                + " will be relaxed in future.\n" //
                + "<p> The order of the created depset should be <i>compatible</i> with the order"
                + " of its <code>transitive</code> depsets. <code>\"default\"</code> order is"
                + " compatible with any other order, all other orders are only compatible with"
                + " themselves.\n" //
                + "<p> Note on backward/forward compatibility. This function currently accepts a"
                + " positional <code>items</code> parameter. It is deprecated and will be removed"
                + " in the future, and after its removal <code>direct</code> will become a sole"
                + " positional parameter of the <code>depset</code> function. Thus, both of the"
                + " following calls are equivalent and future-proof:<br>\n" //
                + "<pre class=language-python>depset(['a', 'b'], transitive = [...])\n" //
                + "depset(direct = ['a', 'b'], transitive = [...])\n" //
                + "</pre>",
        parameters = {
          @Param(
              name = "x",
              type = Object.class,
              defaultValue = "None",
              positional = true,
              named = false,
              noneable = true,
              doc =
                  "A positional parameter distinct from other parameters for legacy support. "
                      + "\n" //
                      + "<p>If <code>--incompatible_disable_depset_inputs</code> is false, this "
                      + "parameter serves as the value of <code>items</code>.</p> "
                      + "\n" //
                      + "<p>If <code>--incompatible_disable_depset_inputs</code> is true, this "
                      + "parameter serves as the value of <code>direct</code>.</p> "
                      + "\n" //
                      + "<p>See the documentation for these parameters for more details."),
          // TODO(cparsons): Make 'order' keyword-only.
          @Param(
              name = "order",
              type = String.class,
              defaultValue = "\"default\"",
              doc =
                  "The traversal strategy for the new depset. See "
                      + "<a href=\"depset.html\">here</a> for the possible values.",
              named = true),
          @Param(
              name = "direct",
              type = Object.class,
              defaultValue = "None",
              positional = false,
              named = true,
              noneable = true,
              doc = "A list of <i>direct</i> elements of a depset. "),
          @Param(
              name = "transitive",
              named = true,
              positional = false,
              type = Sequence.class,
              generic1 = Depset.class,
              noneable = true,
              doc = "A list of depsets whose elements will become indirect elements of the depset.",
              defaultValue = "None"),
          @Param(
              name = "items",
              type = Object.class,
              defaultValue = "[]",
              positional = false,
              doc =
                  "Deprecated: Either an iterable whose items become the direct elements of "
                      + "the new depset, in left-to-right order, or else a depset that becomes "
                      + "a transitive element of the new depset. In the latter case, "
                      + "<code>transitive</code> cannot be specified.",
              disableWithFlag = StarlarkSemantics.FlagIdentifier.INCOMPATIBLE_DISABLE_DEPSET_INPUTS,
              valueWhenDisabled = "[]",
              named = true),
        },
        useStarlarkThread = true)
    public Depset depset(
        Object x,
        String orderString,
        Object direct,
        Object transitive,
        Object items,
        StarlarkThread thread)
        throws EvalException {
      return Depset.depset(x, orderString, direct, transitive, items, thread.getSemantics());
    }

    @SkylarkCallable(
        name = "select",
        doc =
            "<code>select()</code> is the helper function that makes a rule attribute "
                + "<a href=\"$BE_ROOT/common-definitions.html#configurable-attributes\">"
                + "configurable</a>. See "
                + "<a href=\"$BE_ROOT/functions.html#select\">build encyclopedia</a> for details.",
        parameters = {
          @Param(
              name = "x",
              type = Dict.class,
              positional = true,
              doc =
                  "A dict that maps configuration conditions to values. Each key is a label string"
                      + " that identifies a config_setting instance."),
          @Param(
              name = "no_match_error",
              type = String.class,
              defaultValue = "''",
              doc = "Optional custom error to report if no condition matches.",
              named = true),
        })
    public Object select(Dict<?, ?> dict, String noMatchError) throws EvalException {
      return SelectorList.select(dict, noMatchError);
    }

    // TODO(adonovan): move depset here.
  }

  /**
   * A library of Starlark functions (keyed by name) pre-declared in BUILD files. A superset of
   * {@link #COMMON} (e.g. select). Excludes functions in the native module, such as exports_files.
   * Examples: environment_group, select.
   */
  public static final ImmutableMap<String, Object> BUILD = initBUILD();

  private static ImmutableMap<String, Object> initBUILD() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, new BuildLibrary());
    env.putAll(COMMON);
    return env.build();
  }

  @SkylarkGlobalLibrary
  private static class BuildLibrary {
    @SkylarkCallable(
        name = "environment_group",
        doc =
            "Defines a set of related environments that can be tagged onto rules to prevent"
                + "incompatible rules from depending on each other.",
        parameters = {
          @Param(
              name = "name",
              type = String.class,
              positional = false,
              named = true,
              doc = "The name of the rule."),
          // Both parameter below are lists of label designators
          @Param(
              name = "environments",
              type = Sequence.class,
              generic1 = Object.class,
              positional = false,
              named = true,
              doc = "A list of Labels for the environments to be grouped, from the same package."),
          @Param(
              name = "defaults",
              type = Sequence.class,
              generic1 = Object.class,
              positional = false,
              named = true,
              doc = "A list of Labels.")
        }, // TODO(bazel-team): document what that is
        // Not documented by docgen, as this is only available in BUILD files.
        // TODO(cparsons): Devise a solution to document BUILD functions.
        documented = false,
        useStarlarkThread = true)
    public NoneType environmentGroup(
        String name,
        Sequence<?> environmentsList, // <Label>
        Sequence<?> defaultsList, // <Label>
        StarlarkThread thread)
        throws EvalException {
      PackageContext context = getContext(thread);
      List<Label> environments =
          BuildType.LABEL_LIST.convert(
              environmentsList,
              "'environment_group argument'",
              context.pkgBuilder.getBuildFileLabel());
      List<Label> defaults =
          BuildType.LABEL_LIST.convert(
              defaultsList, "'environment_group argument'", context.pkgBuilder.getBuildFileLabel());

      if (environments.isEmpty()) {
        throw Starlark.errorf("environment group %s must contain at least one environment", name);
      }
      try {
        Location loc = thread.getCallerLocation();
        context.pkgBuilder.addEnvironmentGroup(
            name, environments, defaults, context.eventHandler, loc);
        return Starlark.NONE;
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf("environment group has invalid name: %s: %s", name, e.getMessage());
      } catch (Package.NameConflictException e) {
        throw Starlark.errorf("%s", e.getMessage());
      }
    }

    @SkylarkCallable(
        name = "licenses",
        doc = "Declare the license(s) for the code in the current package.",
        parameters = {
          @Param(
              name = "license_strings",
              type = Sequence.class,
              generic1 = String.class,
              doc = "A list of strings, the names of the licenses used.")
        },
        // Not documented by docgen, as this is only available in BUILD files.
        // TODO(cparsons): Devise a solution to document BUILD functions.
        documented = false,
        useStarlarkThread = true)
    public NoneType licenses(
        Sequence<?> licensesList, // list of license strings
        StarlarkThread thread)
        throws EvalException {
      PackageContext context = getContext(thread);
      try {
        License license = BuildType.LICENSE.convert(licensesList, "'licenses' operand");
        context.pkgBuilder.setDefaultLicense(license);
      } catch (ConversionException e) {
        context.eventHandler.handle(Event.error(thread.getCallerLocation(), e.getMessage()));
        context.pkgBuilder.setContainsErrors();
      }
      return Starlark.NONE;
    }

    @SkylarkCallable(
        name = "distribs",
        doc = "Declare the distribution(s) for the code in the current package.",
        parameters = {
          @Param(name = "distribution_strings", type = Object.class, doc = "The distributions.")
        },
        // Not documented by docgen, as this is only available in BUILD files.
        // TODO(cparsons): Devise a solution to document BUILD functions.
        documented = false,
        useStarlarkThread = true)
    public NoneType distribs(Object object, StarlarkThread thread) throws EvalException {
      PackageContext context = getContext(thread);

      try {
        Set<DistributionType> distribs =
            BuildType.DISTRIBUTIONS.convert(object, "'distribs' operand");
        context.pkgBuilder.setDefaultDistribs(distribs);
      } catch (ConversionException e) {
        context.eventHandler.handle(Event.error(thread.getCallerLocation(), e.getMessage()));
        context.pkgBuilder.setContainsErrors();
      }
      return Starlark.NONE;
    }
  }
}
