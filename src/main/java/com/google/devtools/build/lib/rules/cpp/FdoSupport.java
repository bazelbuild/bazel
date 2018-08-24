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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;

/**
 * A container for the path to the FDO profile.
 *
 * <p>{@link FdoSupport} is created from {@link FdoSupportFunction} (a {@link SkyFunction}),
 * which is requested from Skyframe by the {@code cc_toolchain} rule. It's done this way because
 * the path depends on both a command line argument and the location of the workspace and the latter
 * is not available either during configuration creation or during the analysis phase.
 */
@Immutable
@AutoCodec
public final class FdoSupport {
  /**
   * The FDO mode we are operating in.
   */
  public enum FdoMode {
    /** FDO is turned off. */
    OFF,

    /** Profiling-based FDO using an explicitly recorded profile. */
    VANILLA,

    /** FDO based on automatically collected data. */
    AUTO_FDO,

    /** FDO based on cross binary collected data. */
    XBINARY_FDO,

    /** Instrumentation-based FDO implemented on LLVM. */
    LLVM_FDO,
  }

  /**
   * Path of the profile file passed to {@code --fdo_optimize}, or
   * {@code null} if FDO optimization is disabled.  The profile file
   * can be a coverage ZIP or an AutoFDO feedback file.
   */
  // TODO(lberki): This should be a PathFragment.
  // Except that CcProtoProfileProvider#getProfile() calls #exists() on it, which is ridiculously
  // incorrect.
  private final Path fdoProfile;

  /**
   * Creates an FDO support object.
   *
   * @param fdoProfile path to the profile file passed to --fdo_optimize option
   */
  @VisibleForSerialization
  @AutoCodec.Instantiator
  FdoSupport(Path fdoProfile) {
    this.fdoProfile = fdoProfile;
  }

  public Path getFdoProfile() {
    return fdoProfile;
  }
}
