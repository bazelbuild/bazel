// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.docgen.annot.GlobalMethods;
import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.TargetDefinitionContext.NameConflictException;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import java.util.List;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/** A set of miscellaneous APIs that are available to any BUILD file. */
@GlobalMethods(environment = Environment.BUILD)
public class BuildGlobals {

  private BuildGlobals() {}

  public static final BuildGlobals INSTANCE = new BuildGlobals();

  @StarlarkMethod(
      name = "environment_group",
      doc =
          "Defines a set of related environments that can be tagged onto rules to prevent"
              + "incompatible rules from depending on each other.",
      parameters = {
        @Param(name = "name", positional = false, named = true, doc = "The name of the rule."),
        // Both parameter below are lists of label designators
        @Param(
            name = "environments",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = Label.class),
            },
            positional = false,
            named = true,
            doc = "A list of Labels for the environments to be grouped, from the same package."),
        @Param(
            name = "defaults",
            allowedTypes = {
              @ParamType(type = Sequence.class, generic1 = Label.class),
            },
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
    Package.Builder pkgBuilder =
        Package.Builder.fromOrFailAllowBuildOnly(thread, "environment_group()");
    List<Label> environments =
        BuildType.LABEL_LIST.convert(
            environmentsList, "'environment_group argument'", pkgBuilder.getLabelConverter());
    List<Label> defaults =
        BuildType.LABEL_LIST.convert(
            defaultsList, "'environment_group argument'", pkgBuilder.getLabelConverter());

    if (environments.isEmpty()) {
      throw Starlark.errorf("environment group %s must contain at least one environment", name);
    }
    try {
      Location loc = thread.getCallerLocation();
      pkgBuilder.addEnvironmentGroup(
          name, environments, defaults, pkgBuilder.getLocalEventHandler(), loc);
      return Starlark.NONE;
    } catch (LabelSyntaxException e) {
      throw Starlark.errorf("environment group has invalid name: %s: %s", name, e.getMessage());
    } catch (NameConflictException e) {
      throw Starlark.errorf("%s", e.getMessage());
    }
  }

  @StarlarkMethod(
      name = "licenses",
      doc = "Declare the license(s) for the code in the current package.",
      parameters = {
        @Param(
            name = "license_strings",
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
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
    Package.Builder pkgBuilder = Package.Builder.fromOrFailAllowBuildOnly(thread, "licenses()");
    try {
      License license = BuildType.LICENSE.convert(licensesList, "'licenses' operand");
      pkgBuilder.mergePackageArgsFrom(PackageArgs.builder().setLicense(license));
    } catch (ConversionException e) {
      pkgBuilder
          .getLocalEventHandler()
          .handle(
              Package.error(
                  thread.getCallerLocation(), e.getMessage(), Code.LICENSE_PARSE_FAILURE));
      pkgBuilder.setContainsErrors();
    }
    return Starlark.NONE;
  }
}
