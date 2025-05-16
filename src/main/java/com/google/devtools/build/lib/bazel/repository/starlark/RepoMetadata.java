// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.Reproducibility;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.StarlarkValue;

/** The Starlark object returned from a {@code repository_rule}'s implementation function. */
@StarlarkBuiltin(
    name = "repo_metadata",
    category = DocCategory.BUILTIN,
    doc =
        """
        See <a href="repository_ctx#repo_metadata"><code>repository_ctx.repo_metadata</code></a>.
        """)
public record RepoMetadata(Reproducibility reproducible, Dict<?, ?> attrsForReproducibility)
    implements StarlarkValue {
  public static final RepoMetadata NONREPRODUCIBLE =
      new RepoMetadata(Reproducibility.NO, Dict.empty());
}
