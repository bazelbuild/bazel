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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkNativeModuleApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkUtils;
import com.google.devtools.build.lib.syntax.StarlarkThread;

/** The Skylark native module. */
// TODO(laurentlb): Some definitions are duplicated from PackageFactory.
// This class defines:
// native.existing_rule
// native.existing_rules
// native.exports_files -- also global
// native.glob          -- also global
// native.package_group -- also global
// native.package_name
// native.repository_name
//
// PackageFactory also defines:
// distribs            -- hidden?
// licenses            -- hidden?
// package             -- global
// environment_group   -- hidden?
public class SkylarkNativeModule implements SkylarkNativeModuleApi {

  @Override
  public SkylarkList<?> glob(
      SkylarkList<?> include,
      SkylarkList<?> exclude,
      Integer excludeDirectories,
      Object allowEmpty,
      Location loc,
      StarlarkThread thread)
      throws EvalException, ConversionException, InterruptedException {
    SkylarkUtils.checkLoadingPhase(thread, "native.glob", loc);
    try {
      return PackageFactory.callGlob(
          null, include, exclude, excludeDirectories != 0, allowEmpty, loc, thread);
    } catch (IllegalArgumentException e) {
      throw new EvalException(loc, "illegal argument in call to glob", e);
    }
  }

  @Override
  public Object existingRule(String name, Location loc, StarlarkThread thread)
      throws EvalException, InterruptedException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(thread, "native.existing_rule", loc);
    return PackageFactory.callExistingRule(name, loc, thread);
  }

  /*
    If necessary, we could allow filtering by tag (anytag, alltags), name (regexp?), kind ?
    For now, we ignore this, since users can implement it in Skylark.
  */
  @Override
  public SkylarkDict<String, SkylarkDict<String, Object>> existingRules(
      Location loc, StarlarkThread thread) throws EvalException, InterruptedException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(thread, "native.existing_rules", loc);
    return PackageFactory.callExistingRules(loc, thread);
  }

  @Override
  public Runtime.NoneType packageGroup(
      String name,
      SkylarkList<?> packages,
      SkylarkList<?> includes,
      Location loc,
      StarlarkThread thread)
      throws EvalException {
    SkylarkUtils.checkLoadingPhase(thread, "native.package_group", loc);
    return PackageFactory.callPackageFunction(name, packages, includes, loc, thread);
  }

  @Override
  public Runtime.NoneType exportsFiles(
      SkylarkList<?> srcs, Object visibility, Object licenses, Location loc, StarlarkThread thread)
      throws EvalException {
    SkylarkUtils.checkLoadingPhase(thread, "native.exports_files", loc);
    return PackageFactory.callExportsFiles(srcs, visibility, licenses, loc, thread);
  }

  @Override
  public String packageName(Location loc, StarlarkThread thread) throws EvalException {
    SkylarkUtils.checkLoadingPhase(thread, "native.package_name", loc);
    PackageIdentifier packageId =
        PackageFactory.getContext(thread, loc).getBuilder().getPackageIdentifier();
    return packageId.getPackageFragment().getPathString();
  }

  @Override
  public String repositoryName(Location location, StarlarkThread thread) throws EvalException {
    SkylarkUtils.checkLoadingPhase(thread, "native.repository_name", location);
    PackageIdentifier packageId =
        PackageFactory.getContext(thread, location).getBuilder().getPackageIdentifier();
    return packageId.getRepository().toString();
  }
}
