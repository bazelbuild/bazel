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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.authandtls.GoogleAuthUtils;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceGrpcClient;
import java.io.IOException;
import java.util.Set;

/**
 * Bazel's BES module.
 */
public class BazelBuildEventServiceModule
    extends BuildEventServiceModule<BuildEventServiceOptions> {

  @Override
  protected Class<BuildEventServiceOptions> optionsClass() {
    return BuildEventServiceOptions.class;
  }

  @Override
  protected BuildEventServiceClient createBesClient(BuildEventServiceOptions besOptions,
      AuthAndTLSOptions authAndTLSOptions) throws IOException {
    return new BuildEventServiceGrpcClient(
        GoogleAuthUtils.newChannel(besOptions.besBackend, authAndTLSOptions),
        GoogleAuthUtils.newCallCredentials(authAndTLSOptions));
  }

  @Override
  protected Set<String> whitelistedCommands() {
    return ImmutableSet.of("fetch", "build", "test", "run", "query", "coverage", "mobile-install");
  }
}
