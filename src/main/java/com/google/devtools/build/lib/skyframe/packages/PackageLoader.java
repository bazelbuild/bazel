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
package com.google.devtools.build.lib.skyframe.packages;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.ValueOrException;
import java.util.Optional;
import javax.annotation.concurrent.NotThreadSafe;
import net.starlark.java.eval.Module;

/** A standalone library for performing Bazel package loading. */
public interface PackageLoader extends AutoCloseable {
  /**
   * Loads and returns a single package. This method is a simplified shorthand for {@link
   * LoadingContext#loadPackages} when just a single {@link Package} and nothing else is desired.
   */
  Package loadPackage(PackageIdentifier pkgId) throws NoSuchPackageException, InterruptedException;

  /**
   * A reusable context for loading packages and Starlark modules (.bzl or .scl files).
   *
   * <p>Instances of this class are not thread-safe; but the values loaded and returned by its
   * methods are thread-safe. The intention is for LoadingContext to be used e.g. as a local
   * variable in a single-threaded wrapper method of a higher-level abstraction.
   */
  @NotThreadSafe
  interface LoadingContext {
    /**
     * Returns the result of loading a collection of packages. Note that the returned {@link
     * Package}s may contain errors - see {@link Package#containsErrors()} for details.
     *
     * <p>A call to this method clears the list of events accumulated by previous {@link
     * #loadPackages} and {@link #loadModules} calls.
     */
    Result<PackageIdentifier, Package, NoSuchPackageException> loadPackages(
        Iterable<PackageIdentifier> pkgIds) throws InterruptedException;

    /**
     * Returns the result of loading a collection of Starlark modules (i.e. .bzl or .scl files);
     * intended for use by standalone implementations of the starlark_doc_extract rule.
     *
     * <p>A call to this method clears the list of events accumulated by previous {@link
     * #loadPackages} and {@link #loadModules} calls.
     */
    Result<Label, Module, StarlarkModuleLoadingException> loadModules(Iterable<Label> labels)
        throws InterruptedException;

    /**
     * Returns the main repository's repository mapping.
     *
     * <p>Intended to be used for stringifying labels in loaded modules in standalone
     * implementations of the starlark_doc_extract rule.
     */
    RepositoryMapping getRepositoryMapping() throws InterruptedException;
  }

  /** An exception thrown when we fail to load a Starlark module. */
  class StarlarkModuleLoadingException extends Exception {
    StarlarkModuleLoadingException(String message) {
      super(message);
    }

    StarlarkModuleLoadingException(BzlLoadFailedException cause) {
      super(cause);
    }

    /** The {@link FailureDetail} of the underlying exception, if one is available. */
    Optional<FailureDetail> getFailureDetail() {
      if (getCause() instanceof DetailedException) {
        return Optional.of(
            ((DetailedException) getCause()).getDetailedExitCode().getFailureDetail());
      } else {
        return Optional.empty();
      }
    }
  }

  /** Returns a new {@link LoadingContext}. */
  LoadingContext makeLoadingContext() throws InterruptedException;

  /**
   * Shut down the internal threadpools used by the {@link PackageLoader}.
   *
   * <p>Call this method when you are completely done with the {@link PackageLoader} instance,
   * otherwise there may be resource leaks.
   */
  @Override
  void close();

  /**
   * Contains the result of a {@link LoadingContext#loadPackages} or {@link
   * LoadingContext#loadModules} call.
   */
  class Result<K, V, E extends Exception> {
    private final ImmutableMap<K, ValueOrException<V, E>> loadedValues;
    private final ImmutableList<Event> events;

    Result(ImmutableMap<K, ValueOrException<V, E>> loadedValues, ImmutableList<Event> events) {
      this.loadedValues = loadedValues;
      this.events = events;
    }

    /**
     * Returns the map from the requested keys to the corresponding values (packages or modules)
     * which were loaded, or exceptions encountered while attempting to load a given value.
     */
    public ImmutableMap<K, ValueOrException<V, E>> getLoadedValues() {
      return loadedValues;
    }

    /**
     * Returns the events generated by the {@link LoadingContext#loadPackages} or {@link
     * LoadingContext#loadModules} call.
     */
    public ImmutableList<Event> getEvents() {
      return events;
    }
  }
}
