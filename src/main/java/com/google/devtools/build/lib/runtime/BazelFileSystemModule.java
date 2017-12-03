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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.windows.WindowsFileSystem;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;

/**
 * Module to provide a {@link FileSystem} instance that uses {@code SHA256} as the default hash
 * function, or else what's specified by {@code -Dbazel.DigestFunction}.
 *
 * <p>For legacy reasons we can't make the {@link FileSystem} class use {@code SHA256} by default.
 */
public class BazelFileSystemModule extends BlazeModule {

  @Override
  public FileSystem getFileSystem(OptionsProvider startupOptions) throws AbruptExitException {
    final HashFunction hashFunction;
    String value = null;
    try {
      value = System.getProperty("bazel.DigestFunction", "SHA256");
      hashFunction = new HashFunction.Converter().convert(value);
    } catch (OptionsParsingException e) {
      throw new AbruptExitException(
          "The specified hash function '" + value + "' is not supported.",
          ExitCode.COMMAND_LINE_ERROR,
          e);
    }
    if ("0".equals(System.getProperty("io.bazel.EnableJni"))) {
      // Ignore UnixFileSystem, to be used for bootstrapping.
      return OS.getCurrent() == OS.WINDOWS
          ? new WindowsFileSystem(hashFunction)
          : new JavaIoFileSystem(hashFunction);
    }
    // The JNI-based UnixFileSystem is faster, but on Windows it is not available.
    return OS.getCurrent() == OS.WINDOWS
        ? new WindowsFileSystem(hashFunction)
        : new UnixFileSystem(hashFunction);
  }
}
