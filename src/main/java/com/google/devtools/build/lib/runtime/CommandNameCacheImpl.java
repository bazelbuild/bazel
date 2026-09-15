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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.common.options.CommandNameCache;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

class CommandNameCacheImpl implements CommandNameCache {
  private final Map<String, Command> commandMap;
  private final Map<String, ImmutableSet<String>> cache = new HashMap<>();

  CommandNameCacheImpl(Map<String, BlazeCommand> commandMap) {
    // Note: it is important that this map is live, since the commandMap may be altered
    // post-creation.
    this.commandMap =
        Maps.transformValues(
            commandMap, blazeCommand -> blazeCommand.getClass().getAnnotation(Command.class));
  }

  @Override
  public ImmutableSet<String> get(String commandName) {
    ImmutableSet<String> cachedResult = cache.get(commandName);
    if (cachedResult != null) {
      return cachedResult;
    }
    ImmutableSet.Builder<String> builder = ImmutableSet.builder();

    Command command = Preconditions.checkNotNull(commandMap.get(commandName), commandName);
    Set<Command> visited = new HashSet<>();
    visited.add(command);
    Queue<Command> queue = new ArrayDeque<>();
    queue.add(command);
    while (!queue.isEmpty()) {
      Command cur = queue.remove();
      builder.add(cur.name());
      for (Class<? extends BlazeCommand> clazz : cur.inheritsOptionsFrom()) {
        Command parent = clazz.getAnnotation(Command.class);
        if (visited.add(parent)) {
          queue.add(parent);
        }
      }
    }
    cachedResult = builder.build();
    cache.put(commandName, cachedResult);
    return cachedResult;
  }
}
