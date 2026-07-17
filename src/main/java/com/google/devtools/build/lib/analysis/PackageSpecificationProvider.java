// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.starlarkbuildapi.PackageSpecificationProviderApi;
import java.util.Optional;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * A {@link TransitiveInfoProvider} that describes a set of transitive package specifications used
 * in package groups.
 */
public class PackageSpecificationProvider extends NativeInfo
    implements TransitiveInfoProvider, PackageSpecificationProviderApi {

  private static final String STARLARK_NAME = "PackageSpecificationInfo";

  public static final BuiltinProvider<PackageSpecificationProvider> PROVIDER =
      new BuiltinProvider<>(STARLARK_NAME, PackageSpecificationProvider.class) {};

  public static final PackageSpecificationProvider EMPTY =
      new PackageSpecificationProvider(NestedSetBuilder.emptySet(Order.STABLE_ORDER));

  private final NestedSet<PackageGroupContents> packageSpecifications;

  private PackageSpecificationProvider(NestedSet<PackageGroupContents> packageSpecifications) {
    this.packageSpecifications = packageSpecifications;
  }

  /**
   * Creates a {@code PackageSpecificationProvider} by initializing transitive package
   * specifications from {@code targetContext} and {@code packageGroup}.
   */
  public static PackageSpecificationProvider create(
      TargetContext targetContext, PackageGroup packageGroup) {
    return new PackageSpecificationProvider(getPackageSpecifications(targetContext, packageGroup));
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /** Returns set of transitive package specifications used in package groups. */
  public NestedSet<PackageGroupContents> getPackageSpecifications() {
    return packageSpecifications;
  }

  private static NestedSet<PackageGroupContents> getPackageSpecifications(
      TargetContext targetContext, PackageGroup packageGroup) {
    NestedSetBuilder<PackageGroupContents> builder = NestedSetBuilder.stableOrder();
    for (Label includeLabel : packageGroup.getIncludes()) {
      TransitiveInfoCollection include =
          targetContext.findDirectPrerequisite(
              includeLabel, Optional.ofNullable(targetContext.getConfiguration()));
      PackageSpecificationProvider provider = include == null ? null : include.get(PROVIDER);
      if (provider == null) {
        targetContext
            .getAnalysisEnvironment()
            .getEventHandler()
            .handle(
                Event.error(
                    targetContext.getTarget().getLocation(),
                    String.format("Label '%s' does not refer to a package group", includeLabel)));
        continue;
      }

      builder.addTransitive(provider.getPackageSpecifications());
    }

    builder.add(packageGroup.getPackageSpecifications());
    return builder.build();
  }

  @Override
  public boolean targetInAllowlist(Object target) throws EvalException, LabelSyntaxException {
    Label targetLabel;
    if (target instanceof String string) {
      targetLabel = Label.parseCanonical(string);
    } else if (target instanceof Label label) {
      targetLabel = label;
    } else {
      throw Starlark.errorf(
          "expected string or label for 'target' instead of %s", Starlark.type(target));
    }

    return Allowlist.isAvailableFor(packageSpecifications, targetLabel);
  }
}
