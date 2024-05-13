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
import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Doubles;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.worker.WorkerKey;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Iterator;
import java.util.NoSuchElementException;
import javax.annotation.Nullable;

/**
 * Instances of this class represent an estimate of the resource consumption for a particular
 * Action, or the total available resources. We plan to use this to do smarter scheduling of
 * actions, for example making sure that we don't schedule jobs concurrently if they would use so
 * much memory as to cause the machine to thrash.
 */
@Immutable
public class ResourceSet implements ResourceSetOrBuilder {
  public static final String CPU = "cpu";
  public static final String MEMORY = "memory";

  /** For actions that consume negligible resources. */
  public static final ResourceSet ZERO = new ResourceSet(ImmutableMap.of(), 0, null);

  /**
   * Map of extra resources (for example: GPUs, embedded boards, ...) mapping name of the resource
   * to a value.
   */
  private final ImmutableMap<String, Double> resources;

  /** The number of local tests. */
  private final int localTestCount;

  /** The workerKey of used worker. Null if no worker is used. */
  @Nullable private final WorkerKey workerKey;

  private ResourceSet(
      ImmutableMap<String, Double> resources, int localTestCount, @Nullable WorkerKey workerKey) {
    this.resources = resources;
    this.localTestCount = localTestCount;
    this.workerKey = workerKey;
  }

  public static ResourceSet createWithRamCpu(double memoryMb, double cpu) {
    return create(ImmutableMap.of(MEMORY, memoryMb, CPU, cpu));
  }

  public static ResourceSet createWithLocalTestCount(int localTestCount) {
    return create(ImmutableMap.of(), localTestCount);
  }

  public static ResourceSet create(double memoryMb, double cpu, int localTestCount) {
    return create(ImmutableMap.of(MEMORY, memoryMb, CPU, cpu), localTestCount);
  }

  public static ResourceSet create(ImmutableMap<String, Double> resources) {
    return create(resources, 0);
  }

  public static ResourceSet create(ImmutableMap<String, Double> resources, int localTestCount) {
    return create(resources, localTestCount, null);
  }

  public static ResourceSet create(
      ImmutableMap<String, Double> resources, int localTestCount, @Nullable WorkerKey workerKey) {
    return new ResourceSet(resources, localTestCount, workerKey);
  }

  public double get(String resource) {
    return resources.getOrDefault(resource, 0.0);
  }

  public double getMemoryMb() {
    return get(MEMORY);
  }

  public double getCpuUsage() {
    return get(CPU);
  }

  /**
   * Returns the workerKey of worker.
   *
   * <p>If there is no worker requested, then returns null
   */
  public WorkerKey getWorkerKey() {
    return workerKey;
  }

  public ImmutableMap<String, Double> getResources() {
    return resources;
  }

  /** Returns the local test count used. */
  public int getLocalTestCount() {
    return localTestCount;
  }

  @Override
  public String toString() {
    return "Resources: \n"
        + "Memory: "
        + resources.get(MEMORY)
        + "M\n"
        + "CPU: "
        + resources.get(CPU)
        + "\n"
        + resources.entrySet().stream()
            .filter(e -> !e.getKey().equals(CPU) && !e.getKey().equals(MEMORY))
            .collect(
                StringBuilder::new,
                (a, e) -> a.append(e.getKey()).append(": ").append(e.getValue()).append("\n"),
                StringBuilder::append)
        + "Local tests: "
        + localTestCount
        + "\n";
  }

  @Override
  public boolean equals(Object that) {
    if (that == null) {
      return false;
    }

    if (!(that instanceof ResourceSet thatResourceSet)) {
      return false;
    }

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

  /** Converter for {@link ResourceSet}. */
  public static class ResourceSetConverter extends Converter.Contextless<ResourceSet> {
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

  @Override
  public ResourceSet buildResourceSet(OS os, int inputsSize) throws ExecException {
    return this;
  }
}
