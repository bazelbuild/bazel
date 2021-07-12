// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Fake implementation of {@link Registry}, where modules can be freely added and stored in memory.
 */
public class FakeRegistry implements Registry {

  private final String url;
  private final Map<ModuleKey, byte[]> modules = new HashMap<>();

  public FakeRegistry(String url) {
    this.url = url;
  }

  public FakeRegistry addModule(ModuleKey key, String moduleFile) {
    modules.put(key, moduleFile.getBytes(StandardCharsets.UTF_8));
    return this;
  }

  @Override
  public String getUrl() {
    return url;
  }

  @Override
  public Optional<byte[]> getModuleFile(ModuleKey key, ExtendedEventHandler eventHandler) {
    return Optional.ofNullable(modules.get(key));
  }

  @Override
  public RepoSpec getRepoSpec(ModuleKey key, String repoName, ExtendedEventHandler eventHandler) {
    return RepoSpec.builder()
        .setRuleClassName("fake_http_archive_rule")
        .setAttributes(ImmutableMap.of("repo_name", repoName))
        .build();
  }

  /** Fake {@link RegistryFactory} that only supports {@link FakeRegistry}. */
  public static class Factory implements RegistryFactory {

    private int numFakes = 0;
    private final Map<String, FakeRegistry> registries = new HashMap<>();

    public FakeRegistry newFakeRegistry() {
      FakeRegistry registry = new FakeRegistry("fake:" + numFakes++);
      registries.put(registry.getUrl(), registry);
      return registry;
    }

    @Override
    public Registry getRegistryWithUrl(String url) {
      return Preconditions.checkNotNull(registries.get(url), "unknown registry url: %s", url);
    }
  }
}
