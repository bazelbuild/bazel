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
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;

/**
 * Converter to auto-detect the cpu of the machine on which Bazel runs.
 *
 * <p>If the compilation happens remotely then the cpu of the remote machine might be different from
 * the auto-detected one and the --cpu and --host_cpu options must be set explicitly.
 */
public class AutoCpuConverter extends Converter.Contextless<String> {
  @Override
  public String convert(String input) throws OptionsParsingException {
    if (input.isEmpty()) {
      // TODO(philwo) - replace these deprecated names with more logical ones (e.g. k8 becomes
      // linux-x86_64, darwin includes the CPU architecture, ...).
      return switch (OS.getCurrent()) {
        case DARWIN ->
            switch (CPU.getCurrent()) {
              case X86_64 -> "darwin_x86_64";
              case AARCH64 -> "darwin_arm64";
              default -> "unknown";
            };
        case FREEBSD -> "freebsd";
        case OPENBSD -> "openbsd";
        case WINDOWS ->
            switch (CPU.getCurrent()) {
              case X86_64 -> "x64_windows";
              case AARCH64 -> "arm64_windows";
              default -> "unknown";
            };
        case LINUX ->
            switch (CPU.getCurrent()) {
              case X86_32 -> "piii";
              case X86_64 -> "k8";
              case PPC -> "ppc";
              case ARM -> "arm";
              case AARCH64 -> "aarch64";
              case S390X -> "s390x";
              case MIPS64 -> "mips64";
              case RISCV64 -> "riscv64";
              case LOONGARCH64 -> "loongarch64";
              default -> "unknown";
            };
        default -> "unknown";
      };
    }
    return input;
  }

  @Override
  public String getTypeDescription() {
    return "a string";
  }
}
