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
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;

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

  public static Collection<Target> resolvePackageTargets(Package pkg, FilteringPolicy policy) {
    if (policy == FilteringPolicies.NO_FILTER) {
      return pkg.getTargets().values();
    }
    CompactHashSet<Target> builder = CompactHashSet.create();
    for (Target target : pkg.getTargets().values()) {
      if (policy.shouldRetain(target, false)) {
        builder.add(target);
      }
    }
    return builder;
  }

  public static PathFragment getPathFragment(String pathPrefix) throws TargetParsingException {
    PathFragment directory = PathFragment.create(pathPrefix);
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
