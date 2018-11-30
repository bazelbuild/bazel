// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.function.Supplier;

/**
 * Defines a converter for the value of options that take a keyword ("auto" or "HOST_CPUS")
 * optionally followed by an operation ([-|*]<float>), or an integer value.
 */
public class CpuThreadsConverter extends ResourceConverter {

  /**
   * Constructs a CpuThreadsConverter, which evaluates both "auto" and "HOST_CPUS" as max CPU
   * capacity of local host.
   */
  public CpuThreadsConverter() throws OptionsParsingException {
    this(CpuThreadsConverter::getMaxHostValue, CpuThreadsConverter::getMaxHostValue);
  }

  /**
   * Constructs a CpuThreadsConverter. Takes two suppliers, which define the behavior of the "auto"
   * and "HOST_CPUS" keywords.
   *
   * @param autoSupplier a supplier for the value of the "auto" keyword
   * @param hostSupplier a supplier for the value of the "HOST_CPUS" keyword
   */
  public CpuThreadsConverter(Supplier<Integer> autoSupplier, Supplier<Integer> hostSupplier)
      throws OptionsParsingException {
    super(
        ImmutableMap.<String, Supplier<Integer>>builder()
            .put("auto", autoSupplier)
            .put("HOST_CPUS", hostSupplier)
            .build());
  }

  // TODO(jmmv): Using the number of cores has proven to yield reasonable analysis times on
  // Mac Pros and MacBook Pros but we should probably do better than this. (We haven't made
  // any guarantees that "auto" means number of cores precisely to leave us room to tune this
  // further in the future.)
  private static Integer getMaxHostValue() {
    return (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getCpuUsage());
  }

  /**
   * {@inheritDoc}
   *
   * <p>Caps number of threads at 20 for tests.
   *
   * @throws OptionsParsingException if threads < 1
   */
  @Override
  public Integer adjustValue(int threads) throws OptionsParsingException {
    if (System.getenv("TEST_TMPDIR") != null) {
      threads = Math.min(20, threads);
    }
    if (threads < 1) {
      throw new OptionsParsingException(
          String.format("Cpu Threads (%d) must be at least 1.", threads));
    }
    return threads;
  }
}
