// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.ArrayList;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkThread;

/** Context object for a Starlark thread evaluating the VENDOR.bazel file. */
public class VendorThreadContext {

  private final List<RepositoryName> ignoredRepos = new ArrayList<>();
  private final List<RepositoryName> pinnedRepos = new ArrayList<>();

  public static VendorThreadContext fromOrFail(StarlarkThread thread, String what)
      throws EvalException {
    VendorThreadContext context = thread.getThreadLocal(VendorThreadContext.class);
    if (context == null) {
      throw Starlark.errorf("%s can only be called from VENDOR.bazel", what);
    }
    return context;
  }

  public void storeInThread(StarlarkThread thread) {
    thread.setThreadLocal(VendorThreadContext.class, this);
  }

  public VendorThreadContext() {}

  public ImmutableList<RepositoryName> getIgnoredRepos() {
    return ImmutableList.copyOf(ignoredRepos);
  }

  public ImmutableList<RepositoryName> getPinnedRepos() {
    return ImmutableList.copyOf(pinnedRepos);
  }

  public void addIgnoredRepo(RepositoryName repoName) {
    ignoredRepos.add(repoName);
  }

  public void addPinnedRepo(RepositoryName repoName) {
    pinnedRepos.add(repoName);
  }
}
