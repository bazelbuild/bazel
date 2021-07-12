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
import com.google.devtools.build.lib.analysis.configuredtargets.AbstractConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.DefaultInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/** DefaultInfo is provided by all targets implicitly and contains all standard fields. */
@Immutable
public abstract class DefaultInfo extends NativeInfo implements DefaultInfoApi {

  /**
   * Singleton instance of the provider type for {@link DefaultInfo}.
   */
  public static final DefaultInfoProvider PROVIDER = new DefaultInfoProvider();

  @Override
  public DefaultInfoProvider getProvider() {
    return PROVIDER;
  }

  private DefaultInfo() {}

  private DefaultInfo(Location loc) {
    super(loc);
  }

  /**
   * Returns a set of runfiles acting as both the data runfiles and the default runfiles.
   *
   * <p>This is kept for legacy reasons.
   */
  public abstract Runfiles getStatelessRunfiles();

  @Override
  public abstract Runfiles getDataRunfiles();

  @Override
  public abstract Runfiles getDefaultRunfiles();

  /**
   * If the rule producing this info object is marked 'executable' or 'test', this is an artifact
   * representing the file that should be executed to run the target. This is null otherwise.
   */
  public abstract Artifact getExecutable();

  @Override
  public abstract FilesToRunProvider getFilesToRun();

  /** Constructs an optimised DefaultInfo for native targets. */
  public static DefaultInfo build(AbstractConfiguredTarget target) {
    return new DelegatingDefaultInfo(target);
  }

  /** Default implementation of DefaultInfo object for Starlark targets. */
  private static class DefaultDefaultInfo extends DefaultInfo {
    private final Depset files;
    private final Runfiles runfiles;
    private final Runfiles dataRunfiles;
    private final Runfiles defaultRunfiles;
    private final Artifact executable;
    private final FilesToRunProvider filesToRunProvider;

    private DefaultDefaultInfo(
        Location loc,
        Depset files,
        Runfiles runfiles,
        Runfiles dataRunfiles,
        Runfiles defaultRunfiles,
        Artifact executable,
        @Nullable FilesToRunProvider filesToRunProvider) {
      super(loc);
      this.files = files;
      this.runfiles = runfiles;
      this.dataRunfiles = dataRunfiles;
      this.defaultRunfiles = defaultRunfiles;
      this.executable = executable;
      this.filesToRunProvider = filesToRunProvider;
    }

    @Override
    public Depset getFiles() {
      return files;
    }

    @Override
    public FilesToRunProvider getFilesToRun() {
      return filesToRunProvider;
    }

    @Override
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
        // This supports the legacy Starlark runfiles constructor -- if the 'runfiles' attribute
        // is used, then default_runfiles will return all runfiles.
        return runfiles;
      } else {
        return defaultRunfiles;
      }
    }

    @Override
    public Artifact getExecutable() {
      return executable;
    }
  }

  /** Optimised implementation of DefaultInfo object for native targets. */
  private static class DelegatingDefaultInfo extends DefaultInfo {
    private final AbstractConfiguredTarget target;

    DelegatingDefaultInfo(AbstractConfiguredTarget target) {
      this.target = target;
    }

    @Nullable
    @Override
    public Depset getFiles() {
      return Depset.of(Artifact.TYPE, target.getProvider(FileProvider.class).getFilesToBuild());
    }

    @Nullable
    @Override
    public FilesToRunProvider getFilesToRun() {
      return target.getProvider(FilesToRunProvider.class);
    }

    @Nullable
    @Override
    public Runfiles getDataRunfiles() {
      RunfilesProvider runfilesProvider = target.getProvider(RunfilesProvider.class);
      return (runfilesProvider == null) ? Runfiles.EMPTY : runfilesProvider.getDataRunfiles();
    }

    @Nullable
    @Override
    public Runfiles getDefaultRunfiles() {
      RunfilesProvider runfilesProvider = target.getProvider(RunfilesProvider.class);
      return (runfilesProvider == null) ? Runfiles.EMPTY : runfilesProvider.getDefaultRunfiles();
    }

    @Override
    public Runfiles getStatelessRunfiles() {
      return Runfiles.EMPTY;
    }

    @Override
    public Artifact getExecutable() {
      return target.getProvider(FilesToRunProvider.class).getExecutable();
    }
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
    public DefaultInfoApi constructor(
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

      return new DefaultDefaultInfo(
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
