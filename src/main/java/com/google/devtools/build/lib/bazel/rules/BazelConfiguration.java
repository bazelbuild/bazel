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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OptionsUtils.PathFragmentConverter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import java.util.Map;
import javax.annotation.Nullable;

/** Bazel-specific configuration fragment. */
@AutoCodec
@Immutable
public class BazelConfiguration extends Fragment {
  public static final ObjectCodec<BazelConfiguration> CODEC = new BazelConfiguration_AutoCodec();

  /** Command-line options. */
  @AutoCodec(strategy = AutoCodec.Strategy.PUBLIC_FIELDS)
  public static class Options extends FragmentOptions {
    public static final ObjectCodec<Options> CODEC = new BazelConfiguration_Options_AutoCodec();

    @Option(
      name = "experimental_strict_action_env",
      defaultValue = "false",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "If true, Bazel uses an environment with a static value for PATH and does not "
              + "inherit LD_LIBRARY_PATH or TMPDIR. Use --action_env=ENV_VARIABLE if you want to "
              + "inherit specific environment variables from the client, but note that doing so "
              + "can prevent cross-user caching if a shared cache is used."
    )
    public boolean useStrictActionEnv;

    @Option(
      name = "shell_executable",
      converter = PathFragmentConverter.class,
      defaultValue = "null",
      category = "semantics",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      help =
          "Absolute path to the shell executable for Bazel to use. If this is unset, but the "
              + "BAZEL_SH environment variable is set on the first Bazel invocation (that starts "
              + "up a Bazel server), Bazel uses that. If neither is set, Bazel uses a hard-coded "
              + "default path depending on the operating system it runs on (Windows: "
              + "c:/tools/msys64/usr/bin/bash.exe, FreeBSD: /usr/local/bin/bash, all others: "
              + "/bin/bash). Note that using a shell that is not compatible with bash may lead to "
              + "build failures or runtime failures of the generated binaries."
    )
    public PathFragment shellExecutable;

    @Override
    public Options getHost() {
      Options host = (Options) getDefault();
      host.useStrictActionEnv = useStrictActionEnv;
      host.shellExecutable = shellExecutable;
      return host;
    }
  }

  /**
   * Loader for Bazel-specific settings.
   */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new BazelConfiguration(OS.getCurrent(), buildOptions.get(Options.class));
    }

    @Override
    public Class<? extends Fragment> creates() {
      return BazelConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.of(Options.class);
    }
  }

  private static final ImmutableMap<OS, PathFragment> OS_SPECIFIC_SHELL =
      ImmutableMap.<OS, PathFragment>builder()
          .put(OS.WINDOWS, PathFragment.create("c:/tools/msys64/usr/bin/bash.exe"))
          .put(OS.FREEBSD, PathFragment.create("/usr/local/bin/bash"))
          .build();
  private static final PathFragment FALLBACK_SHELL = PathFragment.create("/bin/bash");

  private final OS os;
  private final boolean useStrictActionEnv;
  private final PathFragment shellExecutable;

  public BazelConfiguration(OS os, Options options) {
    this(os, options.useStrictActionEnv, determineShellExecutable(os, options.shellExecutable));
  }

  @AutoCodec.Constructor
  BazelConfiguration(OS os, boolean useStrictActionEnv, PathFragment shellExecutable) {
    this.os = os;
    this.useStrictActionEnv = useStrictActionEnv;
    this.shellExecutable = shellExecutable;
  }

  @Override
  public PathFragment getShellExecutable() {
    return shellExecutable;
  }

  @Override
  public void setupActionEnvironment(Map<String, String> builder) {
    if (useStrictActionEnv) {
      String path = pathOrDefault(os, null, getShellExecutable());
      builder.put("PATH", path);
    } else {
      // TODO(ulfjack): Avoid using System.getenv; it's the wrong environment!
      builder.put("PATH", pathOrDefault(os, System.getenv("PATH"), getShellExecutable()));

      // TODO(laszlocsomor): Remove setting TMP/TEMP here, and set a meaningful value just before
      // executing the action.
      // Setting TMP=null, TEMP=null has the effect of copying the client's TMP/TEMP to the action's
      // environment. This is a short-term workaround to get temp-requiring actions working on
      // Windows. Its detrimental effect is that the client's TMP/TEMP becomes part of the actions's
      // key. Yet, we need this for now to build Android programs, because the Android BusyBox is
      // written in Java and tries to create temp directories using
      // java.nio.file.Files.createTempDirectory, which needs TMP or TEMP (or USERPROFILE) on
      // Windows, otherwise they return c:\windows which is non-writable.
      builder.put("TMP", null);
      builder.put("TEMP", null);

      String ldLibraryPath = System.getenv("LD_LIBRARY_PATH");
      if (ldLibraryPath != null) {
        builder.put("LD_LIBRARY_PATH", ldLibraryPath);
      }

      String tmpdir = System.getenv("TMPDIR");
      if (tmpdir != null) {
        builder.put("TMPDIR", tmpdir);
      }
    }
  }

  private static PathFragment determineShellExecutable(OS os, PathFragment fromOption) {
    if (fromOption != null) {
      return fromOption;
    }
    // Honor BAZEL_SH env variable for backwards compatibility.
    String path = System.getenv("BAZEL_SH");
    if (path != null) {
      return PathFragment.create(path);
    }
    // TODO(ulfjack): instead of using the OS Bazel runs on, we need to use the exec platform, which
    // may be different for remote execution. For now, this can be overridden with
    // --shell_executable, so at least there's a workaround.
    PathFragment result = OS_SPECIFIC_SHELL.get(os);
    return result != null ? result : FALLBACK_SHELL;
  }

  @VisibleForTesting
  static String pathOrDefault(OS os, @Nullable String path, @Nullable PathFragment sh) {
    // TODO(ulfjack): The default PATH should be set from the exec platform, which may be different
    // from the local machine. For now, this can be overridden with --action_env=PATH=<value>, so
    // at least there's a workaround.
    if (os != OS.WINDOWS) {
      return path == null ? "/bin:/usr/bin" : path;
    }

    // Attempt to compute the MSYS root (the real Windows path of "/") from `sh`.
    if (sh != null && sh.getParentDirectory() != null) {
      String newPath = sh.getParentDirectory().getPathString();
      if (sh.getParentDirectory().endsWith(PathFragment.create("usr/bin"))) {
        newPath +=
            ";" + sh.getParentDirectory().getParentDirectory().replaceName("bin").getPathString();
      } else if (sh.getParentDirectory().endsWith(PathFragment.create("bin"))) {
        newPath +=
            ";" + sh.getParentDirectory().replaceName("usr").getRelative("bin").getPathString();
      }
      newPath = newPath.replace('/', '\\');

      if (path != null) {
        newPath += ";" + path;
      }
      return newPath;
    } else if (path != null) {
      return path;
    } else {
      return "";
    }
  }
}
