// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org /licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleDynamicFrameworkInfoApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Provider containing information about an Apple dynamic framework. This provider contains:
 *
 * <ul>
 *   <li>'framework_dirs': The framework path names used as link inputs in order to link against the
 *       dynamic framework
 *   <li>'framework_files': The full set of artifacts that should be included as inputs to link
 *       against the dynamic framework
 *   <li>'binary': The dylib binary artifact of the dynamic framework
 *   <li>'objc': An {@link ObjcProvider} which contains information about the transitive
 *       dependencies linked into the binary, (intended so that bundle loaders depending on this
 *       executable may avoid relinking symbols included in the loadable binary
 * </ul>
 */
@Immutable
public final class AppleDynamicFrameworkInfo extends NativeInfo
    implements AppleDynamicFrameworkInfoApi<PathFragment, Artifact> {

  /** Skylark name for the AppleDynamicFrameworkInfo. */
  public static final String SKYLARK_NAME = "AppleDynamicFramework";

  /** Skylark constructor and identifier for AppleDynamicFrameworkInfo. */
  public static final NativeProvider<AppleDynamicFrameworkInfo> SKYLARK_CONSTRUCTOR =
      new NativeProvider<AppleDynamicFrameworkInfo>(
          AppleDynamicFrameworkInfo.class, SKYLARK_NAME) {};

  /** Field name for the dylib binary artifact of the dynamic framework. */
  public static final String DYLIB_BINARY_FIELD_NAME = "binary";
  /** Field name for the framework path names of the dynamic framework. */
  public static final String FRAMEWORK_DIRS_FIELD_NAME = "framework_dirs";
  /** Field name for the framework link-input artifacts of the dynamic framework. */
  public static final String FRAMEWORK_FILES_FIELD_NAME = "framework_files";
  /** Field name for the {@link ObjcProvider} containing dependency information. */
  public static final String OBJC_PROVIDER_FIELD_NAME = "objc";

  private final NestedSet<PathFragment> dynamicFrameworkDirs;
  private final NestedSet<Artifact> dynamicFrameworkFiles;
  private final @Nullable Artifact dylibBinary;
  private final ObjcProvider depsObjcProvider;

  public AppleDynamicFrameworkInfo(@Nullable Artifact dylibBinary,
      ObjcProvider depsObjcProvider,
      NestedSet<PathFragment> dynamicFrameworkDirs,
      NestedSet<Artifact> dynamicFrameworkFiles) {
    super(SKYLARK_CONSTRUCTOR);
    this.dylibBinary = dylibBinary;
    this.depsObjcProvider = depsObjcProvider;
    this.dynamicFrameworkDirs = dynamicFrameworkDirs;
    this.dynamicFrameworkFiles = dynamicFrameworkFiles;
  }

  @Override
  public NestedSet<PathFragment> getDynamicFrameworkDirs() {
    return dynamicFrameworkDirs;
  }

  @Override
  public NestedSet<Artifact> getDynamicFrameworkFiles() {
    return dynamicFrameworkFiles;
  }

  @Override
  public Artifact getAppleDylibBinary() {
    return dylibBinary;
  }

  @Override
  public ObjcProvider getDepsObjcProvider() {
    return depsObjcProvider;
  }
}