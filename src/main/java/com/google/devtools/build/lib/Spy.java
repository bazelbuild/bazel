// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.syntax.LoadStatement;
import com.google.devtools.build.lib.syntax.LoadStatement.Binding;
import java.net.URL;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Spy {
  public static final Spy INSTANCE = new Spy();

  private final Set<String> reposInLoads;
  // symbol to repo name
  private final Map<String, String> repoSymbols;
  private final Map<String, List<URL>> reposToUrls;
  private final Map<String, String> repoDefs;
  private final Map<String, List<String>> repoInits;

  private Spy() {
    reposInLoads = Sets.newConcurrentHashSet();
    reposToUrls = Maps.newConcurrentMap();
    repoSymbols = Maps.newConcurrentMap();
    repoDefs = Maps.newConcurrentMap();
    repoInits = Maps.newConcurrentMap();
  }

  public void onEvalLoadStatement(LoadStatement loadStatement) {
    String repo = getLoadsRepo(loadStatement);
    if (repo != null) {
      reposInLoads.add(repo);
    }
  }

  private static String getLoadsRepo(LoadStatement loadStatement) {
    String fileLabel = loadStatement.getImport().getValue().trim();
    if (fileLabel.startsWith("@")) {
      int idx = fileLabel.indexOf('/');
      if (idx < 0) {
        System.out.println("Can not parse load: " + fileLabel);
        return null;
      }
      return fileLabel.substring(1, idx);
    }
    return null;
  }

  public void onDownloadingSomething(String repo, List<URL> urls) {
    reposToUrls.computeIfAbsent(repo, k -> Collections.synchronizedList(Lists.newArrayList()))
        .addAll(urls);
  }

  public void onWorkspaceChunkLoading(
      List<LoadStatement> loadStatements,
      Map<String, String> repoDefinitions,
      Map<String, String> expressionStatements) {
    for (LoadStatement loadStatement : loadStatements) {
      String repo = getLoadsRepo(loadStatement);
      if (repo == null) {
        continue;
      }

      ImmutableList<Binding> bindings = loadStatement.getBindings();
      for (Binding binding : bindings) {
        repoSymbols.put(binding.getLocalName().getName(), repo);
      }
    }
    repoDefs.putAll(repoDefinitions);
    for (Map.Entry<String, String> entry : expressionStatements.entrySet()) {
      String repo = repoSymbols.get(entry.getKey());
      if (repo != null) {
        repoInits.computeIfAbsent(repo, k -> Collections.synchronizedList(Lists.newArrayList()))
            .add(entry.getValue());
      }
    }
  }

  public Set<String> getReposInLoads() {
    return reposInLoads;
  }

  public Map<String, List<URL>> getReposToUrls() {
    return reposToUrls;
  }

  public Map<String, String> getRepoSymbols() {
    return repoSymbols;
  }

  public Map<String, String> getRepoDefs() {
    return repoDefs;
  }

  public Map<String, List<String>> getRepoInits() {
    return repoInits;
  }
}
