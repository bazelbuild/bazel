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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Utility class encapsulating the standard definition of the {@code package()} function of BUILD
 * files.
 */
public class PackageCallable {

  protected PackageCallable() {}

  public static final PackageCallable INSTANCE = new PackageCallable();

  @StarlarkMethod(
      name = "package",
      documented = false, // documented in docgen/templates/be/functions.vm
      extraKeywords = @Param(name = "kwargs", defaultValue = "{}"),
      useStarlarkThread = true)
  public Object packageCallable(Map<String, Object> kwargs, StarlarkThread thread)
      throws EvalException {
    Package.Builder pkgBuilder = PackageFactory.getContext(thread).pkgBuilder;
    if (pkgBuilder.isPackageFunctionUsed()) {
      throw new EvalException("'package' can only be used once per BUILD file");
    }
    pkgBuilder.setPackageFunctionUsed();

    if (kwargs.isEmpty()) {
      throw new EvalException("at least one argument must be given to the 'package' function");
    }

    Location loc = thread.getCallerLocation();
    for (Map.Entry<String, Object> kwarg : kwargs.entrySet()) {
      String name = kwarg.getKey();
      Object rawValue = kwarg.getValue();
      processParam(name, rawValue, pkgBuilder, loc);
    }
    return Starlark.NONE;
  }

  /**
   * Handles one parameter. Subclasses can add new parameters by overriding this method and falling
   * back on the super method when the parameter does not match.
   */
  protected void processParam(
      String name, Object rawValue, Package.Builder pkgBuilder, Location loc) throws EvalException {
    if (name.equals("default_visibility")) {
      List<Label> value = convert(BuildType.LABEL_LIST, rawValue, pkgBuilder);
      pkgBuilder.setDefaultVisibility(RuleVisibility.parse(value));

    } else if (name.equals("default_testonly")) {
      Boolean value = convert(Type.BOOLEAN, rawValue, pkgBuilder);
      pkgBuilder.setDefaultTestonly(value);

    } else if (name.equals("default_deprecation")) {
      String value = convert(Type.STRING, rawValue, pkgBuilder);
      pkgBuilder.setDefaultDeprecation(value);

    } else if (name.equals("features")) {
      List<String> value = convert(Type.STRING_LIST, rawValue, pkgBuilder);
      pkgBuilder.addFeatures(value);

    } else if (name.equals("licenses")) {
      License value = convert(BuildType.LICENSE, rawValue, pkgBuilder);
      pkgBuilder.setDefaultLicense(value);

    } else if (name.equals("distribs")) {
      Set<DistributionType> value = convert(BuildType.DISTRIBUTIONS, rawValue, pkgBuilder);
      pkgBuilder.setDefaultDistribs(value);

    } else if (name.equals("default_compatible_with")) {
      List<Label> value = convert(BuildType.LABEL_LIST, rawValue, pkgBuilder);
      pkgBuilder.setDefaultCompatibleWith(value, name, loc);

    } else if (name.equals("default_restricted_to")) {
      List<Label> value = convert(BuildType.LABEL_LIST, rawValue, pkgBuilder);
      pkgBuilder.setDefaultRestrictedTo(value, name, loc);

    } else if (name.equals("default_applicable_licenses")) {
      List<Label> value = convert(BuildType.LABEL_LIST, rawValue, pkgBuilder);
      if (!pkgBuilder.getDefaultPackageMetadata().isEmpty()) {
        pkgBuilder.addEvent(
            Package.error(
                loc,
                "Can not set both default_package_metadata and default_applicable_licenses."
                    + " Move all declarations to default_package_metadata.",
                Code.INVALID_PACKAGE_SPECIFICATION));
      }
      pkgBuilder.setDefaultPackageMetadata(value, name, loc);

    } else if (name.equals("default_package_metadata")) {
      List<Label> value = convert(BuildType.LABEL_LIST, rawValue, pkgBuilder);
      if (!pkgBuilder.getDefaultPackageMetadata().isEmpty()) {
        pkgBuilder.addEvent(
            Package.error(
                loc,
                "Can not set both default_package_metadata and default_applicable_licenses."
                    + " Move all declarations to default_package_metadata.",
                Code.INVALID_PACKAGE_SPECIFICATION));
      }
      pkgBuilder.setDefaultPackageMetadata(value, name, loc);

    } else {
      throw Starlark.errorf("unexpected keyword argument: %s", name);
    }
  }

  protected static <T> T convert(Type<T> type, Object value, Package.Builder pkgBuilder)
      throws EvalException {
    return type.convert(value, "'package' argument", pkgBuilder.getLabelConverter());
  }
}
