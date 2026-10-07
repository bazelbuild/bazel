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
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.starlarkbuildapi.PackageSpecificationProviderApi;
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
   * Creates a {@code PackageSpecificationProvider} from the given transitive package
   * specifications.
   */
  public static PackageSpecificationProvider create(
      NestedSet<PackageGroupContents> packageSpecifications) {
    return new PackageSpecificationProvider(packageSpecifications);
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /** Returns set of transitive package specifications used in package groups. */
  public NestedSet<PackageGroupContents> getPackageSpecifications() {
    return packageSpecifications;
  }

  /** Returns true if the given label's package is included in the package specifications. */
  public static boolean isAvailableFor(
      NestedSet<PackageGroupContents> packageGroupContents, Label relevantLabel) {
    return packageGroupContents.toList().stream()
        .anyMatch(p -> p.containsPackage(relevantLabel.getPackageIdentifier()));
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

    return isAvailableFor(packageSpecifications, targetLabel);
  }
}
