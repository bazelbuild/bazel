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
package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPatternResolver;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Common utility methods for target pattern resolution.
 */
public final class TargetPatternResolverUtil {
  private TargetPatternResolverUtil() {
    // Utility class.
  }

  public static String getParsingErrorMessage(String message, String originalPattern) {
    if (originalPattern == null) {
      return message;
    } else {
      return String.format("while parsing '%s': %s", originalPattern, message);
    }
  }

  public static ResolvedTargets<Target> resolvePackageTargets(Package pkg,
                                                              FilteringPolicy policy) {
    ResolvedTargets.Builder<Target> builder = ResolvedTargets.builder();
    for (Target target : pkg.getTargets()) {
      if (policy.shouldRetain(target, false)) {
        builder.add(target);
      }
    }
    return builder.build();
  }

  public static void validatePatternPackage(String originalPattern,
      PathFragment packageNameFragment, TargetPatternResolver<?> resolver)
      throws TargetParsingException {
    String packageName = packageNameFragment.toString();
    // It's possible for this check to pass, but for
    // Label.validatePackageNameFull to report an error because the
    // package name is illegal.  That's a little weird, but we can live with
    // that for now--see test case: testBadPackageNameButGoodEnoughForALabel.
    if (LabelValidator.validatePackageName(packageName) != null) {
      throw new TargetParsingException("'" + packageName + "' is not a valid package name");
    }
    if (!resolver.isPackage(PackageIdentifier.createInDefaultRepo(packageName))) {
      throw new TargetParsingException(
          TargetPatternResolverUtil.getParsingErrorMessage(
              "no such package '" + packageName + "': BUILD file not found on package path",
              originalPattern));
    }
  }

  public static PathFragment getPathFragment(String pathPrefix) throws TargetParsingException {
    PathFragment directory = new PathFragment(pathPrefix);
    if (directory.containsUplevelReferences()) {
      throw new TargetParsingException("up-level references are not permitted: '"
          + directory.getPathString() + "'");
    }
    if (!pathPrefix.isEmpty() && (LabelValidator.validatePackageName(pathPrefix) != null)) {
      throw new TargetParsingException("'" + pathPrefix + "' is not a valid package name");
    }
    return directory;
  }
}
