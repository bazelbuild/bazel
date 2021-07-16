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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Fake implementation of {@link Registry}, where modules can be freely added and stored in memory.
 */
public class FakeRegistry implements Registry {

  private final String url;
  private final Map<ModuleKey, String> modules = new HashMap<>();

  public FakeRegistry(String url) {
    this.url = url;
  }

  public FakeRegistry addModule(ModuleKey key, String moduleFile) {
    modules.put(key, moduleFile);
    return this;
  }

  @Override
  public String getUrl() {
    return url;
  }

  @Override
  public Optional<byte[]> getModuleFile(ModuleKey key, ExtendedEventHandler eventHandler) {
    return Optional.ofNullable(modules.get(key)).map(value -> value.getBytes(UTF_8));
  }

  @Override
  public RepoSpec getRepoSpec(ModuleKey key, String repoName, ExtendedEventHandler eventHandler) {
    return RepoSpec.builder()
        .setRuleClassName("fake_http_archive_rule")
        .setAttributes(ImmutableMap.of("repo_name", repoName))
        .build();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof FakeRegistry
        && this.url.equals(((FakeRegistry) other).url)
        && this.modules.equals(((FakeRegistry) other).modules);
  }

  @Override
  public int hashCode() {
    return Objects.hash(url, modules);
  }

  public static final Factory DEFAULT_FACTORY = new Factory();

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
