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

package com.google.devtools.build.lib.buildeventservice;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.authentication.TlsOptions;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventservice.client.ManagedBuildEventServiceGrpcClient;
import com.google.devtools.build.lib.grpc.GrpcUtils;
import com.google.devtools.build.lib.runtime.AuthHeadersProvider;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Bazel's BES module.
 */
public class BazelBuildEventServiceModule
    extends BuildEventServiceModule<BuildEventServiceOptions> {

  @AutoValue
  abstract static class BackendConfig {
    abstract String besBackend();

    abstract TlsOptions tlsOptions();

    abstract AuthHeadersProvider authHeadersProvider();
  }

  private BuildEventServiceClient client;
  private BackendConfig config;

  @Override
  protected Class<BuildEventServiceOptions> optionsClass() {
    return BuildEventServiceOptions.class;
  }

  @Override
  protected BuildEventServiceClient getBesClient(BuildEventServiceOptions besOptions,
      TlsOptions tlsOptions, Map<String, AuthHeadersProvider> authHeadersProvidersMap)
      throws AbruptExitException {
    AuthHeadersProvider authHeadersProvider = selectAuthHeadersProvider(authHeadersProvidersMap);
    BackendConfig newConfig =
        new AutoValue_BazelBuildEventServiceModule_BackendConfig(
            besOptions.besBackend, tlsOptions, authHeadersProvider);
    if (client == null || !Objects.equals(config, newConfig)) {
      clearBesClient();
      config = newConfig;
      client = new ManagedBuildEventServiceGrpcClient(
          GrpcUtils.newManagedChannel(besOptions.besBackend, ImmutableList.of(),
              tlsOptions.tlsEnabled, tlsOptions.tlsAuthorityOverride, tlsOptions.tlsCertificate),
              GrpcUtils.newCallCredentials(authHeadersProvider));
    }
    return client;
  }

  @Override
  protected void clearBesClient() {
    if (client != null) {
      client.shutdown();
    }
    this.client = null;
    this.config = null;
  }

  private static final ImmutableSet<String> WHITELISTED_COMMANDS =
      ImmutableSet.of(
          "fetch",
          "build",
          "test",
          "run",
          "query",
          "aquery",
          "cquery",
          "coverage",
          "mobile-install");

  @Override
  protected Set<String> whitelistedCommands() {
    return WHITELISTED_COMMANDS;
  }

  @Nullable
  private static AuthHeadersProvider selectAuthHeadersProvider(
      Map<String, AuthHeadersProvider> authHeadersProvidersMap) {
    // TODO(buchgr): Implement a selection strategy based on name.
    for (AuthHeadersProvider provider : authHeadersProvidersMap.values()) {
      if (provider.isEnabled()) {
        return provider;
      }
    }

    return null;
  }
}
