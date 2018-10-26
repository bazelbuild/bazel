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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.config.PackageProviderForConfigurations;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.SkyframePackageLoader;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;

/**
 * Repeats functionality of {@link SkyframePackageLoader} but uses
 * {@link SkyFunction.Environment#getValue} instead of {@link MemoizingEvaluator#evaluate}
 * for node evaluation
 */
class SkyframePackageLoaderWithValueEnvironment implements PackageProviderForConfigurations {
  private final SkyFunction.Environment env;

  SkyframePackageLoaderWithValueEnvironment(SkyFunction.Environment env) {
    this.env = env;
  }

  @Override
  public ExtendedEventHandler getEventHandler() {
    return env.getListener();
  }

  private Package getPackage(final PackageIdentifier pkgIdentifier)
      throws NoSuchPackageException, InterruptedException {
    SkyKey key = PackageValue.key(pkgIdentifier);
    PackageValue value = (PackageValue) env.getValueOrThrow(key, NoSuchPackageException.class);
    if (value != null) {
      return value.getPackage();
    }
    return null;
  }

  @Override
  public Target getTarget(Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException {
    Package pkg = getPackage(label.getPackageIdentifier());
    return pkg == null ? null : pkg.getTarget(label.getName());
  }

  @Override
  public void addDependency(Package pkg, String fileName)
      throws LabelSyntaxException, IOException, InterruptedException {
    RootedPath fileRootedPath = RootedPath.toRootedPath(pkg.getSourceRoot(),
        pkg.getPackageIdentifier().getSourceRoot().getRelative(fileName));
    FileValue result = (FileValue) env.getValue(FileValue.key(fileRootedPath));
    if (result != null && !result.exists()) {
      throw new IOException();
    }
  }

  @Override
  public boolean valuesMissing() {
    return env.valuesMissing();
  }
}
