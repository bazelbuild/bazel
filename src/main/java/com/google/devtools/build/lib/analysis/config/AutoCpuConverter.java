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

package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.util.CPU;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * Converter to auto-detect the cpu of the machine on which Bazel runs.
 *
 * <p>If the compilation happens remotely then the cpu of the remote machine might be different from
 * the auto-detected one and the --cpu and --host_cpu options must be set explicitly.
 */
public class AutoCpuConverter implements Converter<String> {
  @Override
  public String convert(String input) throws OptionsParsingException {
    if (input.isEmpty()) {
      // TODO(philwo) - replace these deprecated names with more logical ones (e.g. k8 becomes
      // linux-x86_64, darwin includes the CPU architecture, ...).
      switch (OS.getCurrent()) {
        case DARWIN:
          return "darwin";
        case FREEBSD:
          return "freebsd";
        case OPENBSD:
          return "openbsd";
        case WINDOWS:
          switch (CPU.getCurrent()) {
            case X86_64:
              return "x64_windows";
            default:
              // We only support x64 Windows for now.
              return "unknown";
          }
        case LINUX:
          switch (CPU.getCurrent()) {
            case X86_32:
              return "piii";
            case X86_64:
              return "k8";
            case PPC:
              return "ppc";
            case ARM:
              return "arm";
            case AARCH64:
              return "aarch64";
            case S390X:
              return "s390x";
            default:
              return "unknown";
          }
        default:
          return "unknown";
      }
    }
    return input;
  }

  /** Reverses the conversion performed by {@link #convert} to return the matching OS, CPU pair. */
  public static Pair<CPU, OS> reverse(String input) {
    if (input == null || input.length() == 0 || "unknown".equals(input)) {
      // Use the auto-detected values.
      return Pair.of(CPU.getCurrent(), OS.getCurrent());
    }

    // Handle the easy cases.
    if (input.startsWith("darwin")) {
      return Pair.of(CPU.getCurrent(), OS.DARWIN);
    } else if (input.startsWith("freebsd")) {
      return Pair.of(CPU.getCurrent(), OS.FREEBSD);
    } else if (input.startsWith("openbsd")) {
      return Pair.of(CPU.getCurrent(), OS.OPENBSD);
    } else if (input.startsWith("x64_windows")) {
      return Pair.of(CPU.getCurrent(), OS.WINDOWS);
    }

    // Handle the Linux cases.
    if (input.equals("piii")) {
      return Pair.of(CPU.X86_32, OS.LINUX);
    } else if (input.equals("k8")) {
      return Pair.of(CPU.X86_64, OS.LINUX);
    } else if (input.equals("ppc")) {
      return Pair.of(CPU.PPC, OS.LINUX);
    } else if (input.equals("arm")) {
      return Pair.of(CPU.ARM, OS.LINUX);
    } else if (input.equals("s390x")) {
      return Pair.of(CPU.S390X, OS.LINUX);
    }

    // Use the auto-detected values.
    return Pair.of(CPU.getCurrent(), OS.getCurrent());
  }

  @Override
  public String getTypeDescription() {
    return "a string";
  }
}
