// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.downloader.Downloader;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.query2.QueryEnvironmentFactory;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.query.output.OutputFormatter;
import com.google.devtools.build.lib.runtime.commands.InfoItem;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import java.util.function.Supplier;

/**
 * Builder class to create a {@link BlazeRuntime} instance. This class is part of the module API,
 * which allows modules to affect how the server is initialized.
 */
public final class ServerBuilder {
  private QueryEnvironmentFactory queryEnvironmentFactory;
  private final InvocationPolicy.Builder invocationPolicyBuilder = InvocationPolicy.newBuilder();
  private final ImmutableList.Builder<BlazeCommand> commands = ImmutableList.builder();
  private final ImmutableMap.Builder<String, InfoItem> infoItems = ImmutableMap.builder();
  private final ImmutableList.Builder<QueryFunction> queryFunctions = ImmutableList.builder();
  private final ImmutableList.Builder<OutputFormatter> queryOutputFormatters =
      ImmutableList.builder();
  private final ImmutableList.Builder<PackageFactory.EnvironmentExtension> environmentExtensions =
      ImmutableList.builder();
  private final BuildEventArtifactUploaderFactoryMap.Builder buildEventArtifactUploaderFactories =
      new BuildEventArtifactUploaderFactoryMap.Builder();
  private final ImmutableMap.Builder<String, AuthHeadersProvider> authHeadersProvidersMap =
      ImmutableMap.builder();
  private RepositoryRemoteExecutorFactory repositoryRemoteExecutorFactory;
  private Supplier<Downloader> downloaderSupplier = () -> null;

  @VisibleForTesting
  public ServerBuilder() {}

  QueryEnvironmentFactory getQueryEnvironmentFactory() {
    return queryEnvironmentFactory == null
        ? new QueryEnvironmentFactory()
        : queryEnvironmentFactory;
  }

  InvocationPolicy getInvocationPolicy() {
    return invocationPolicyBuilder.build();
  }

  ImmutableMap<String, InfoItem> getInfoItems() {
    return infoItems.build();
  }

  ImmutableList<QueryFunction> getQueryFunctions() {
    return queryFunctions.build();
  }

  ImmutableList<OutputFormatter> getQueryOutputFormatters() {
    return queryOutputFormatters.build();
  }

  // Visible for WorkspaceResolver.
  public ImmutableList<PackageFactory.EnvironmentExtension> getEnvironmentExtensions() {
    return environmentExtensions.build();
  }

  @VisibleForTesting
  public ImmutableList<BlazeCommand> getCommands() {
    return commands.build();
  }

  public BuildEventArtifactUploaderFactoryMap getBuildEventArtifactUploaderMap() {
    return buildEventArtifactUploaderFactories.build();
  }

  public RepositoryRemoteExecutorFactory getRepositoryRemoteExecutorFactory() {
    return repositoryRemoteExecutorFactory;
  }

  public Supplier<Downloader> getDownloaderSupplier() {
    return downloaderSupplier;
  }

  /**
   * Merges the given invocation policy into the per-server invocation policy. While this can accept
   * any number of policies, the end result is order-dependent if multiple policies attempt to
   * police the same options, so it's probably a good idea to not have too many modules that call
   * this.
   */
  public ServerBuilder addInvocationPolicy(InvocationPolicy policy) {
    invocationPolicyBuilder.mergeFrom(Preconditions.checkNotNull(policy));
    return this;
  }

  /**
   * Sets a factory for creating {@link
   * com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment} instances. Note that
   * only one factory per server is allowed. If none is set, the server uses the default
   * implementation.
   */
  public ServerBuilder setQueryEnvironmentFactory(QueryEnvironmentFactory queryEnvironmentFactory) {
    Preconditions.checkState(
        this.queryEnvironmentFactory == null,
        "At most one query environment factory supported. But found two: %s and %s",
        this.queryEnvironmentFactory,
        queryEnvironmentFactory);
    this.queryEnvironmentFactory = Preconditions.checkNotNull(queryEnvironmentFactory);
    return this;
  }

  /**
   * Adds the given command to the server. This overload only exists to avoid array object creation
   * in the common case.
   */
  public ServerBuilder addCommands(BlazeCommand command) {
    this.commands.add(Preconditions.checkNotNull(command));
    return this;
  }

  /** Adds the given commands to the server. */
  public ServerBuilder addCommands(BlazeCommand... commands) {
    this.commands.add(commands);
    return this;
  }

  /**
   * Adds the given items as info items to the info command. It is an error to add info items with
   * the same name to the same builder, regardless of whether that happens within the same module or
   * across modules.
   */
  public ServerBuilder addInfoItems(InfoItem... infoItems) {
    for (InfoItem item : infoItems) {
      this.infoItems.put(item.getName(), item);
    }
    return this;
  }

  public ServerBuilder addQueryFunctions(QueryFunction... functions) {
    this.queryFunctions.add(functions);
    return this;
  }

  public ServerBuilder addQueryOutputFormatters(OutputFormatter... formatters) {
    this.queryOutputFormatters.add(formatters);
    return this;
  }

  public ServerBuilder addQueryOutputFormatters(Iterable<OutputFormatter> formatters) {
    this.queryOutputFormatters.addAll(formatters);
    return this;
  }

  public ServerBuilder addEnvironmentExtension(PackageFactory.EnvironmentExtension extension) {
    this.environmentExtensions.add(extension);
    return this;
  }

  public ServerBuilder addBuildEventArtifactUploaderFactory(
      BuildEventArtifactUploaderFactory uploaderFactory, String name) {
    buildEventArtifactUploaderFactories.add(name, uploaderFactory);
    return this;
  }

  public ServerBuilder setRepositoryRemoteExecutorFactory(
      RepositoryRemoteExecutorFactory repositoryRemoteExecutorFactory) {
    this.repositoryRemoteExecutorFactory = repositoryRemoteExecutorFactory;
    return this;
  }

  public ServerBuilder setDownloaderSupplier(Supplier<Downloader> downloaderSupplier) {
    this.downloaderSupplier = downloaderSupplier;
    return this;
  }

  /**
   * Register a provider of authentication headers that blaze modules can use. See {@link
   * AuthHeadersProvider} for more details.
   */
  public ServerBuilder addAuthHeadersProvider(
      String name, AuthHeadersProvider authHeadersProvider) {
    authHeadersProvidersMap.put(name, authHeadersProvider);
    return this;
  }

  /** Returns a map of all registered {@link AuthHeadersProvider}s. */
  public ImmutableMap<String, AuthHeadersProvider> getAuthHeadersProvidersMap() {
    return authHeadersProvidersMap.build();
  }
}
