// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skylarkbuildapi.DefaultInfoApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import javax.annotation.Nullable;

/** DefaultInfo is provided by all targets implicitly and contains all standard fields. */
@Immutable
public final class DefaultInfo extends NativeInfo implements DefaultInfoApi {

  private final Depset files;
  private final Runfiles runfiles;
  private final Runfiles dataRunfiles;
  private final Runfiles defaultRunfiles;
  private final Artifact executable;
  private final FilesToRunProvider filesToRunProvider;

  /**
   * Singleton instance of the provider type for {@link DefaultInfo}.
   */
  public static final DefaultInfoProvider PROVIDER = new DefaultInfoProvider();

  private DefaultInfo(
      @Nullable RunfilesProvider runfilesProvider,
      FileProvider fileProvider,
      FilesToRunProvider filesToRunProvider) {
    this(
        Location.BUILTIN,
        Depset.of(Artifact.TYPE, fileProvider.getFilesToBuild()),
        Runfiles.EMPTY,
        (runfilesProvider == null) ? Runfiles.EMPTY : runfilesProvider.getDataRunfiles(),
        (runfilesProvider == null) ? Runfiles.EMPTY : runfilesProvider.getDefaultRunfiles(),
        filesToRunProvider.getExecutable(),
        filesToRunProvider);
  }

  private DefaultInfo(
      Location loc,
      Depset files,
      Runfiles runfiles,
      Runfiles dataRunfiles,
      Runfiles defaultRunfiles,
      Artifact executable,
      @Nullable FilesToRunProvider filesToRunProvider) {
    super(PROVIDER, loc);
    this.files = files;
    this.runfiles = runfiles;
    this.dataRunfiles = dataRunfiles;
    this.defaultRunfiles = defaultRunfiles;
    this.executable = executable;
    this.filesToRunProvider = filesToRunProvider;
  }

  public static DefaultInfo build(
      @Nullable RunfilesProvider runfilesProvider,
      FileProvider fileProvider,
      FilesToRunProvider filesToRunProvider) {
    return new DefaultInfo(runfilesProvider, fileProvider, filesToRunProvider);
  }

  @Override
  public Depset getFiles() {
    return files;
  }

  @Override
  public FilesToRunProvider getFilesToRun() {
    return filesToRunProvider;
  }

  /**
   * Returns a set of runfiles acting as both the data runfiles and the default runfiles.
   *
   * This is kept for legacy reasons.
   */
  public Runfiles getStatelessRunfiles() {
    return runfiles;
  }

  @Override
  public Runfiles getDataRunfiles() {
    return dataRunfiles;
  }

  @Override
  public Runfiles getDefaultRunfiles() {
    if (dataRunfiles == null && defaultRunfiles == null) {
      // This supports the legacy skylark runfiles constructor -- if the 'runfiles' attribute
      // is used, then default_runfiles will return all runfiles.
      return runfiles;
    } else {
      return defaultRunfiles;
    }
  }

  /**
   * If the rule producing this info object is marked 'executable' or 'test', this is an artifact
   * representing the file that should be executed to run the target. This is null otherwise.
   */
  public Artifact getExecutable() {
    return executable;
  }

  /**
   * Provider implementation for {@link DefaultInfoApi}.
   */
  public static class DefaultInfoProvider extends BuiltinProvider<DefaultInfo>
      implements DefaultInfoApi.DefaultInfoApiProvider<Runfiles, Artifact> {
    private DefaultInfoProvider() {
      super("DefaultInfo", DefaultInfo.class);
    }

    @Override
    public DefaultInfo constructor(
        Object files,
        Object runfilesObj,
        Object dataRunfilesObj,
        Object defaultRunfilesObj,
        Object executable,
        StarlarkThread thread)
        throws EvalException {

      Runfiles statelessRunfiles = castNoneToNull(Runfiles.class, runfilesObj);
      Runfiles dataRunfiles = castNoneToNull(Runfiles.class, dataRunfilesObj);
      Runfiles defaultRunfiles = castNoneToNull(Runfiles.class, defaultRunfilesObj);

      if ((statelessRunfiles != null) && (dataRunfiles != null || defaultRunfiles != null)) {
        throw Starlark.errorf(
            "Cannot specify the provider 'runfiles' together with 'data_runfiles' or"
                + " 'default_runfiles'");
      }

      return new DefaultInfo(
          thread.getCallerLocation(),
          castNoneToNull(Depset.class, files),
          statelessRunfiles,
          dataRunfiles,
          defaultRunfiles,
          castNoneToNull(Artifact.class, executable),
          null);
    }
  }

  private static <T> T castNoneToNull(Class<T> clazz, Object value) {
    if (value == Starlark.NONE) {
      return null;
    } else {
      return clazz.cast(value);
    }
  }
}
