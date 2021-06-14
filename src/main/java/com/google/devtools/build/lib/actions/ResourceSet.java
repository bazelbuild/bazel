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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Splitter;
import com.google.common.primitives.Doubles;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Instances of this class represent an estimate of the resource consumption for a particular
 * Action, or the total available resources. We plan to use this to do smarter scheduling of
 * actions, for example making sure that we don't schedule jobs concurrently if they would use so
 * much memory as to cause the machine to thrash.
 */
@Immutable
@AutoCodec
public class ResourceSet {

  /** For actions that consume negligible resources. */
  public static final ResourceSet ZERO = new ResourceSet(0.0, 0.0, 0);

  /** The amount of real memory (resident set size). */
  private final double memoryMb;

  /** The number of CPUs, or fractions thereof. */
  private final double cpuUsage;

  /** The number of local tests. */
  private final int localTestCount;

  private ResourceSet(double memoryMb, double cpuUsage, int localTestCount) {
    this.memoryMb = memoryMb;
    this.cpuUsage = cpuUsage;
    this.localTestCount = localTestCount;
  }

  /**
   * Returns a new ResourceSet with the provided values for memoryMb and cpuUsage, and with 0.0 for
   * ioUsage and localTestCount. Use this method in action resource definitions when they aren't
   * local tests.
   */
  public static ResourceSet createWithRamCpu(double memoryMb, double cpuUsage) {
    if (memoryMb == 0 && cpuUsage == 0) {
      return ZERO;
    }
    return new ResourceSet(memoryMb, cpuUsage, 0);
  }

  /**
   * Returns a new ResourceSet with the provided value for localTestCount, and 0.0 for memoryMb,
   * cpuUsage, and ioUsage. Use this method in action resource definitions when they are local tests
   * that acquire no local resources.
   */
  public static ResourceSet createWithLocalTestCount(int localTestCount) {
    return new ResourceSet(0.0, 0.0, localTestCount);
  }

  /**
   * Returns a new ResourceSet with the provided values for memoryMb, cpuUsage, ioUsage, and
   * localTestCount. Most action resource definitions should use {@link #createWithRamCpu} or
   * {@link #createWithLocalTestCount(int)}. Use this method primarily when constructing
   * ResourceSets that represent available resources.
   */
  @AutoCodec.Instantiator
  public static ResourceSet create(
      double memoryMb, double cpuUsage, int localTestCount) {
    if (memoryMb == 0 && cpuUsage == 0 && localTestCount == 0) {
      return ZERO;
    }
    return new ResourceSet(memoryMb, cpuUsage, localTestCount);
  }

  /** Returns the amount of real memory (resident set size) used in MB. */
  public double getMemoryMb() {
    return memoryMb;
  }

  /**
   * Returns the number of CPUs (or fractions thereof) used.
   * For a CPU-bound single-threaded process, this will be 1.0.
   * For a single-threaded process which spends part of its
   * time waiting for I/O, this will be somewhere between 0.0 and 1.0.
   * For a multi-threaded or multi-process application,
   * this may be more than 1.0.
   */
  public double getCpuUsage() {
    return cpuUsage;
  }

  /** Returns the local test count used. */
  public int getLocalTestCount() {
    return localTestCount;
  }

  @Override
  public String toString() {
    return "Resources: \n"
        + "Memory: " + memoryMb + "M\n"
        + "CPU: " + cpuUsage + "\n"
        + "Local tests: " + localTestCount + "\n";
  }

  @Override
  public boolean equals(Object that) {
    if (that == null) {
      return false;
    }

    if (!(that instanceof ResourceSet)) {
      return false;
    }

    ResourceSet thatResourceSet = (ResourceSet) that;
    return thatResourceSet.getMemoryMb() == getMemoryMb()
        && thatResourceSet.getCpuUsage() == getCpuUsage()
        && thatResourceSet.localTestCount == getLocalTestCount();
  }

  @Override
  public int hashCode() {
    int p = 239;
    return Doubles.hashCode(getMemoryMb())
        + Doubles.hashCode(getCpuUsage()) * p
        + getLocalTestCount() * p * p;
  }

  public static class ResourceSetConverter implements Converter<ResourceSet> {
    private static final Splitter SPLITTER = Splitter.on(',');

    @Override
    public ResourceSet convert(String input) throws OptionsParsingException {
      Iterator<String> values = SPLITTER.split(input).iterator();
      try {
        double memoryMb = Double.parseDouble(values.next());
        double cpuUsage = Double.parseDouble(values.next());
        // There used to be a third field here called ioUsage. In order to not break existing users,
        // we keep expecting a third field, which must be a double. In the future, we may accept the
        // two-param variant, and then even phase out the three-param variant.
        Double.parseDouble(values.next());
        if (values.hasNext()) {
          throw new OptionsParsingException("Expected exactly 3 comma-separated float values");
        }
        if (memoryMb <= 0.0 || cpuUsage <= 0.0) {
          throw new OptionsParsingException("All resource values must be positive");
        }
        return create(memoryMb, cpuUsage, Integer.MAX_VALUE);
      } catch (NumberFormatException | NoSuchElementException nfe) {
        throw new OptionsParsingException("Expected exactly 3 comma-separated float values", nfe);
      }
    }

    @Override
    public String getTypeDescription() {
      return "comma-separated available amount of RAM (in MB), CPU (in cores) and "
          + "available I/O (1.0 being average workstation)";
    }
  }
}
