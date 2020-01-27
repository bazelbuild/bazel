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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.util.List;
import java.util.Set;

/** A global library of Starlark functions which are available only when evaluating BUILD files. */
@SkylarkGlobalLibrary
class StarlarkBuildLibrary {

  /**
   * Map of Starlark values (keyed by their symbol name) defined by this global library for use in
   * the global Starlark environment for BUILD files.
   */
  public static final ImmutableMap<String, Object> BINDINGS = initializeBindings();

  private static ImmutableMap<String, Object> initializeBindings() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    Starlark.addMethods(env, new StarlarkBuildLibrary());
    return env.build();
  }

  private StarlarkBuildLibrary() {
    // Not instantiable outside this class.
  }

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
