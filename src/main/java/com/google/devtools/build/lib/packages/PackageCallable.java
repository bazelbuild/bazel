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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Location;

/**
 * Utility class encapsulating the definition of the {@code package()} function of BUILD files.
 *
 * <p>Also includes the definitions of those arguments to {@code package()} that are available in
 * all Bazel environments.
 */
public class PackageCallable {

  private PackageCallable() {}

  /**
   * Returns a {@link StarlarkCallable} that implements the {@code package()} functionality.
   *
   * @param packageArgs a list of {@link PackageArgument}s to support, beyond the standard ones
   *     included in every Bazel environment
   */
  // TODO(b/280446865): Consider eliminating the package() extensibility mechanism altogether.
  // There is currently only one use case: distinguishing the set of package() arguments available
  // in OSS Bazel vs internally to Google. Instead of registering these arguments and passing them
  // to this factory method to obtain a package() callable, we could instead define two
  // @StarlarkMethod-annotated Java functions implementing the two versions of package(), and
  // register the appropriate one with the ConfiguredRuleClassProvider.Builder.
  public static StarlarkCallable newPackageCallable(List<PackageArgument<?>> packageArgs) {
    // Construct a map of PackageArguments, which the returned callable closes over.
    ImmutableMap.Builder<String, PackageArgument<?>> argsBuilder = ImmutableMap.builder();
    for (PackageArgument<?> arg : getCommonArguments()) {
      argsBuilder.put(arg.getName(), arg);
    }
    for (PackageArgument<?> arg : packageArgs) {
      argsBuilder.put(arg.getName(), arg);
    }
    final ImmutableMap<String, PackageArgument<?>> packageArguments = argsBuilder.buildOrThrow();

    return new StarlarkCallable() {
      @Override
      public String getName() {
        return "package";
      }

      @Override
      public String toString() {
        return "package(...)";
      }

      @Override
      public boolean isImmutable() {
        return true;
      }

      @Override
      public void repr(Printer printer) {
        printer.append("<built-in function package>");
      }

      @Override
      public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
          throws EvalException {
        if (!args.isEmpty()) {
          throw new EvalException("unexpected positional arguments");
        }
        Package.Builder pkgBuilder = PackageFactory.getContext(thread).pkgBuilder;

        // Validate parameter list
        if (pkgBuilder.isPackageFunctionUsed()) {
          throw new EvalException("'package' can only be used once per BUILD file");
        }
        pkgBuilder.setPackageFunctionUsed();

        // Each supplied argument must name a PackageArgument.
        if (kwargs.isEmpty()) {
          throw new EvalException("at least one argument must be given to the 'package' function");
        }
        Location loc = thread.getCallerLocation();
        for (Map.Entry<String, Object> kwarg : kwargs.entrySet()) {
          String name = kwarg.getKey();
          PackageArgument<?> pkgarg = packageArguments.get(name);
          if (pkgarg == null) {
            throw Starlark.errorf("unexpected keyword argument: %s", name);
          }
          pkgarg.convertAndProcess(pkgBuilder, loc, kwarg.getValue());
        }
        return Starlark.NONE;
      }
    };
  }

  /** Returns the basic set of {@link PackageArgument}s. */
  private static ImmutableList<PackageArgument<?>> getCommonArguments() {
    return ImmutableList.of(
        new DefaultDeprecation(),
        new DefaultDistribs(),
        new DefaultApplicableLicenses(),
        new DefaultPackageMetadata(),
        new DefaultLicenses(),
        new DefaultTestOnly(),
        new DefaultVisibility(),
        new Features(),
        new DefaultCompatibleWith(),
        new DefaultRestrictedTo());
  }

  private static class DefaultVisibility extends PackageArgument<List<Label>> {
    private DefaultVisibility() {
      super("default_visibility", BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, List<Label> value)
        throws EvalException {
      pkgBuilder.setDefaultVisibility(RuleVisibility.parse(value));
    }
  }

  private static class DefaultTestOnly extends PackageArgument<Boolean> {
    private DefaultTestOnly() {
      super("default_testonly", Type.BOOLEAN);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, Boolean value) {
      pkgBuilder.setDefaultTestonly(value);
    }
  }

  private static class DefaultDeprecation extends PackageArgument<String> {
    private DefaultDeprecation() {
      super("default_deprecation", Type.STRING);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, String value) {
      pkgBuilder.setDefaultDeprecation(value);
    }
  }

  private static class Features extends PackageArgument<List<String>> {
    private Features() {
      super("features", Type.STRING_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, List<String> value) {
      pkgBuilder.addFeatures(value);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#APPLICABLE_LICENSES_ATTR} when not explicitly
   * specified.
   */
  private static class DefaultApplicableLicenses extends PackageArgument<List<Label>> {
    private DefaultApplicableLicenses() {
      super("default_applicable_licenses", BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, List<Label> value) {
      if (!pkgBuilder.getDefaultPackageMetadata().isEmpty()) {
        pkgBuilder.addEvent(
            Package.error(
                location,
                "Can not set both default_package_metadata and default_applicable_licenses."
                    + " Move all declarations to default_package_metadata.",
                Code.INVALID_PACKAGE_SPECIFICATION));
      }

      pkgBuilder.setDefaultPackageMetadata(value, "default_package_metadata", location);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#APPLICABLE_LICENSES_ATTR} when not explicitly
   * specified.
   */
  private static class DefaultPackageMetadata extends PackageArgument<List<Label>> {
    private static final String DEFAULT_PACKAGE_METADATA_ATTRIBUTE = "default_package_metadata";

    private DefaultPackageMetadata() {
      super(DEFAULT_PACKAGE_METADATA_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, List<Label> value) {
      if (!pkgBuilder.getDefaultPackageMetadata().isEmpty()) {
        pkgBuilder.addEvent(
            Package.error(
                location,
                "Can not set both default_package_metadata and default_applicable_licenses."
                    + " Move all declarations to default_package_metadata.",
                Code.INVALID_PACKAGE_SPECIFICATION));
      }
      pkgBuilder.setDefaultPackageMetadata(value, DEFAULT_PACKAGE_METADATA_ATTRIBUTE, location);
    }
  }

  private static class DefaultLicenses extends PackageArgument<License> {
    private DefaultLicenses() {
      super("licenses", BuildType.LICENSE);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, License value) {
      pkgBuilder.setDefaultLicense(value);
    }
  }

  private static class DefaultDistribs extends PackageArgument<Set<DistributionType>> {
    private DefaultDistribs() {
      super("distribs", BuildType.DISTRIBUTIONS);
    }

    @Override
    protected void process(
        Package.Builder pkgBuilder, Location location, Set<DistributionType> value) {
      pkgBuilder.setDefaultDistribs(value);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#COMPATIBLE_ENVIRONMENT_ATTR} when not
   * explicitly specified.
   */
  private static class DefaultCompatibleWith extends PackageArgument<List<Label>> {
    private static final String DEFAULT_COMPATIBLE_WITH_ATTRIBUTE = "default_compatible_with";

    private DefaultCompatibleWith() {
      super(DEFAULT_COMPATIBLE_WITH_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, List<Label> value) {
      pkgBuilder.setDefaultCompatibleWith(value, DEFAULT_COMPATIBLE_WITH_ATTRIBUTE, location);
    }
  }

  /**
   * Declares the package() attribute specifying the default value for {@link
   * com.google.devtools.build.lib.packages.RuleClass#RESTRICTED_ENVIRONMENT_ATTR} when not
   * explicitly specified.
   */
  private static class DefaultRestrictedTo extends PackageArgument<List<Label>> {
    private static final String DEFAULT_RESTRICTED_TO_ATTRIBUTE = "default_restricted_to";

    private DefaultRestrictedTo() {
      super(DEFAULT_RESTRICTED_TO_ATTRIBUTE, BuildType.LABEL_LIST);
    }

    @Override
    protected void process(Package.Builder pkgBuilder, Location location, List<Label> value) {
      pkgBuilder.setDefaultRestrictedTo(value, DEFAULT_RESTRICTED_TO_ATTRIBUTE, location);
    }
  }
}
