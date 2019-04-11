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
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkNativeModuleApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkUtils;
import com.google.devtools.build.lib.syntax.Type.ConversionException;

/**
 * A class for the Skylark native module. TODO(laurentlb): Some definitions are duplicated from
 * PackageFactory.
 */
public class SkylarkNativeModule implements SkylarkNativeModuleApi {

  @Override
  public SkylarkList<?> glob(
      SkylarkList<?> include,
      SkylarkList<?> exclude,
      Integer excludeDirectories,
      FuncallExpression ast,
      Environment env)
      throws EvalException, ConversionException, InterruptedException {
    SkylarkUtils.checkLoadingPhase(env, "native.glob", ast.getLocation());
    return PackageFactory.callGlob(null, include, exclude, excludeDirectories != 0, ast, env);
  }

  @Override
  public Object existingRule(String name, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "native.existing_rule", ast.getLocation());
    return PackageFactory.callExistingRule(name, ast, env);
  }

  /*
    If necessary, we could allow filtering by tag (anytag, alltags), name (regexp?), kind ?
    For now, we ignore this, since users can implement it in Skylark.
  */
  @Override
  public SkylarkDict<String, SkylarkDict<String, Object>> existingRules(
      FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    SkylarkUtils.checkLoadingOrWorkspacePhase(env, "native.existing_rules", ast.getLocation());
    return PackageFactory.callExistingRules(ast, env);
  }

  @Override
  public Runtime.NoneType packageGroup(String name, SkylarkList<?> packages,
      SkylarkList<?> includes,
      FuncallExpression ast, Environment env) throws EvalException {
    SkylarkUtils.checkLoadingPhase(env, "native.package_group", ast.getLocation());
    return PackageFactory.callPackageFunction(name, packages, includes, ast, env);
  }

  @Override
  public Runtime.NoneType exportsFiles(SkylarkList<?> srcs, Object visibility, Object licenses,
      FuncallExpression ast, Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingPhase(env, "native.exports_files", ast.getLocation());
    return PackageFactory.callExportsFiles(srcs, visibility, licenses, ast, env);
  }

  @Override
  public String packageName(FuncallExpression ast, Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingPhase(env, "native.package_name", ast.getLocation());
    PackageIdentifier packageId =
        PackageFactory.getContext(env, ast.getLocation()).getBuilder().getPackageIdentifier();
    return packageId.getPackageFragment().getPathString();
  }

  @Override
  public String repositoryName(Location location, Environment env)
      throws EvalException {
    SkylarkUtils.checkLoadingPhase(env, "native.repository_name", location);
    PackageIdentifier packageId =
        PackageFactory.getContext(env, location).getBuilder().getPackageIdentifier();
    return packageId.getRepository().toString();
  }
}
