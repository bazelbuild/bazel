// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Utility methods for use by ConfiguredTarget implementations.
 */
public abstract class Util {

  private Util() {}

  //---------- Label and Target related methods

  /**
   * Returns the workspace-relative path of the specified target (file or rule).
   *
   * <p>For example, "//foo/bar:wiz" and "//foo:bar/wiz" both result in "foo/bar/wiz".
   */
  public static PathFragment getWorkspaceRelativePath(Target target) {
    return getWorkspaceRelativePath(target.getLabel());
  }

  /**
   * Returns the workspace-relative path of the specified target (file or rule).
   *
   * <p>For example, "//foo/bar:wiz" and "//foo:bar/wiz" both result in "foo/bar/wiz".
   */
  public static PathFragment getWorkspaceRelativePath(Label label) {
    return label.getPackageFragment().getRelative(label.getName());
  }

  /**
   * Returns the workspace-relative path of the specified target (file or rule),
   * prepending a prefix and appending a suffix.
   *
   * <p>For example, "//foo/bar:wiz" and "//foo:bar/wiz" both result in "foo/bar/wiz".
   */
  public static PathFragment getWorkspaceRelativePath(Target target, String prefix, String suffix) {
    return target.getLabel().getPackageFragment().getRelative(prefix + target.getName() + suffix);
  }

  /**
   * Checks if a PathFragment contains a '-'.
   */
  public static boolean containsHyphen(PathFragment path) {
    return path.getPathString().indexOf('-') >= 0;
  }
}
