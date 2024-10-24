# Copyright 2023 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# WARNING:
# https://github.com/bazelbuild/bazel/issues/17713
# .bzl files in this package (tools/build_defs/repo) are evaluated
# in a Starlark environment without "@_builtins" injection, and must not refer
# to symbols associated with build/workspace .bzl files

"""Returns the default canonical id to use for downloads."""

visibility("public")

DEFAULT_CANONICAL_ID_ENV = "BAZEL_HTTP_RULES_URLS_AS_DEFAULT_CANONICAL_ID"

CANONICAL_ID_DOC = """A canonical ID of the file downloaded.

If specified and non-empty, Bazel will not take the file from cache, unless it
was added to the cache by a request with the same canonical ID.

If unspecified or empty, Bazel by default uses the URLs of the file as the
canonical ID. This helps catch the common mistake of updating the URLs without
also updating the hash, resulting in builds that succeed locally but fail on
machines without the file in the cache. This behavior can be disabled with
--repo_env={env}=0.
""".format(env = DEFAULT_CANONICAL_ID_ENV)

def get_default_canonical_id(repository_ctx, urls):
    """Returns the default canonical id to use for downloads.

    Returns `""` (empty string) when Bazel is run with
    `--repo_env=BAZEL_HTTP_RULES_URLS_AS_DEFAULT_CANONICAL_ID=0`.

    e.g.
    ```python
    load("@bazel_tools//tools/build_defs/repo:cache.bzl", "get_default_canonical_id")
    # ...
        repository_ctx.download_and_extract(
            url = urls,
            integrity = integrity
            canonical_id = get_default_canonical_id(repository_ctx, urls),
        ),
    ```

    Args:
      repository_ctx: The repository context of the repository rule calling this utility
        function.
      urls: A list of URLs matching what is passed to `repository_ctx.download` and
        `repository_ctx.download_and_extract`.
    """
    if repository_ctx.os.environ.get(DEFAULT_CANONICAL_ID_ENV) == "0":
        return ""

    # Do not sort URLs to prevent the following scenario:
    # 1. http_archive with urls = [B, A] created.
    # 2. Successful fetch from B results in canonical ID "A B".
    # 3. Order of urls is flipped to [A, B].
    # 4. Fetch would reuse cache entry for "A B", even though A may be broken (it has never been
    #    fetched before).
    return " ".join(urls)
