# Copyright 2022 The Bazel Authors. All rights reserved.
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
import os
import re
import shutil
import sys

from absl import app

_LINK_PATTERN = re.compile(
    r'^(\s*<link rel="canonical" href=")(/versions/[^/]+/)([^"]*)(">)$',
    re.MULTILINE)
_ROOT = "https://bazel.build/"
_REDIRECTS = {
    "":
        "",
    "skylark/repository_rules":
        "rules/repository_rules",
    "glossary":
        "reference/glossary",
    "install-redhat":
        "install/redhat",
    "skylark/tutorial-sharing-variables":
        "rules/tutorial-sharing-variables",
    "install-suse":
        "install/suse",
    "aquery":
        "docs/aquery",
    "migrate-cocoapods":
        "migrate/cocoapods",
    "build-event-protocol":
        "docs/build-event-protocol",
    "releases":
        "release",
    "migrate-xcode":
        "migrate/xcode",
    "coverage":
        "docs/coverage",
    "workspace-log":
        "docs/workspace-log",
    "bazel-and-android":
        "docs/bazel-and-android",
    "windows":
        "docs/windows",
    "sandboxing":
        "docs/sandboxing",
    "persistent-workers":
        "docs/persistent-workers",
    "skylark/build-style":
        "rules/build-style",
    "skylark/macros":
        "rules/macros",
    "skylark/tutorial-creating-a-macro":
        "rules/tutorial-creating-a-macro",
    "creating-workers":
        "docs/creating-workers",
    "dynamic-execution":
        "docs/dynamic-execution",
    "install":
        "docs/mobile-install",
    "hermeticity":
        "concepts/hermeticity",
    "bazel-and-apple":
        "docs/bazel-and-apple",
    "install-os-x":
        "install/os-x",
    "configurable-attributes":
        "docs/configurable-attributes",
    "android-ndk":
        "docs/android-ndk",
    "skylark/bzl-style":
        "rules/bzl-style",
    "skylark/windows_tips":
        "rules/windows_tips",
    "query-how-to":
        "docs/query-how-to",
    "build-javascript":
        "",
    "tutorial/ios-app":
        "tutorials/ios-app",
    "skylark/aspects":
        "rules/aspects",
    "remote-execution-caching-debug":
        "docs/remote-execution-caching-debug",
    "tutorial/cpp":
        "docs/bazel-and-cpp",
    "build-ref":
        "concepts/build-ref",
    "skylark/testing":
        "rules/testing",
    "getting-started":
        "contribute/getting-started",
    "skylark/tutorial-custom-verbs":
        "rules/tutorial-custom-verbs",
    "skylark/rules":
        "rules/rules",
    "skylark/errors/read-only-variable":
        "rules/errors/read-only-variable",
    "memory-saving-mode":
        "docs/memory-saving-mode",
    "command-line-reference":
        "reference/command-line-reference",
    "toolchain_resolution_implementation":
        "docs/toolchain_resolution_implementation",
    "generate-workspace":
        "",
    "best-practices":
        "docs/best-practices",
    "cquery":
        "docs/cquery",
    "platforms":
        "docs/platforms",
    "skylark/performance":
        "rules/performance",
    "platforms-intro":
        "concepts/platforms-intro",
    "exec-groups":
        "reference/exec-groups",
    "backward-compatibility":
        "release/backward-compatibility",
    "skylark/concepts":
        "rules/concepts",
    "bep-examples":
        "docs/bep-examples",
    "multiplex-worker":
        "docs/multiplex-worker",
    "bazel-and-cpp":
        "docs/bazel-and-cpp",
    "query":
        "reference/query",
    "bazel-and-java":
        "docs/bazel-and-java",
    "ide":
        "contribute/guide",
    "guide":
        "contribute/guide",
    "skylark/depsets":
        "rules/depsets",
    "bazel-container":
        "docs/bazel-container",
    "install-bazelisk":
        "install/bazelisk",
    "bep-glossary":
        "docs/bep-glossary",
    "rule-challenges":
        "docs/rule-challenges",
    "tutorial/java":
        "docs/bazel-and-java",
    "skylark/language":
        "rules/language",
    "tutorial/cc-toolchain-config":
        "tutorials/cc-toolchain-config",
    "visibility":
        "concepts/visibility",
    "install-windows":
        "install/windows",
    "android-instrumentation-test":
        "docs/android-instrumentation-test",
    "tutorial/android-app":
        "tutorials/android-app",
    "remote-caching":
        "docs/remote-caching",
    "migrate-maven":
        "migrate/maven",
    "output_directories":
        "docs/output_directories",
    "external":
        "docs/external",
    "remote-execution-sandbox":
        "docs/remote-execution-sandbox",
    "bazel-and-javascript":
        "docs/bazel-and-javascript",
    "bazel-vision":
        "start/bazel-vision",
    "integrating-with-rules-cc":
        "docs/integrating-with-rules-cc",
    "bzlmod":
        "docs/bzlmod",
    "user-manual":
        "docs/user-manual",
    "cpp-use-cases":
        "tutorials/cpp-use-cases",
    "mobile-install":
        "docs/mobile-install",
    "android-build-performance":
        "docs/android-build-performance",
    "test-encyclopedia":
        "reference/test-encyclopedia",
    "remote-execution-ci":
        "docs/remote-execution-ci",
    "install-compile-source":
        "install/compile-source",
    "toolchains":
        "docs/toolchains",
    "bazel-overview":
        "start/bazel-intro",
    "completion":
        "install/completion",
    "remote-execution-rules":
        "docs/remote-execution-rules",
    "install-ubuntu":
        "install/ubuntu",
    "skylark/config":
        "rules/config",
    "skylark/faq":
        "rules/faq",
    "cc-toolchain-config-reference":
        "docs/cc-toolchain-config-reference",
    "versioning":
        "release/versioning",
    "remote-execution":
        "docs/remote-execution",
    "skylark/rules-tutorial":
        "rules/rules-tutorial",
    "updating-bazel":
        "versions/updating-bazel",
    "remote-caching-debug":
        "docs/remote-caching-debug",
    "skylark/deploying":
        "rules/deploying",
    "rules":
        "rules"
}


def maybe_create_dir(dest):
  dir_path = os.path.dirname(dest)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)


def transform(src, dest):
  with open(src, "rt", encoding="utf-8") as f:
    content = f.read()

  fixed_content = _LINK_PATTERN.sub(repl, content, count=1)
  if 'link rel="canonical"' in content and fixed_content == content:
    print('[W] Failed to transform {}.'.format(src), file=sys.stderr)

  with open(dest, "wt", encoding="utf-8") as f:
    f.write(fixed_content)


def repl(m):
  return "{}{}{}{}".format(
      m.group(1), _ROOT, get_new_page(m.group(3)), m.group(4))


def get_new_page(old):
  raw = removesuffix(old, ".html")
  if raw.startswith("skylark/lib"):
    return raw.replace("skylark/lib", "rules/lib")
  elif raw.startswith("be/"):
    return "reference/" + raw
  elif raw.startswith("repo/"):
    return "rules/lib/" + raw

  return _REDIRECTS.get(raw, "")


def removesuffix(full_str, suffix):
  if not full_str.endswith(suffix):
    return full_str

  return full_str[:-len(suffix)]


def removeprefix(full_str, prefix):
  if not full_str.startswith(prefix):
    return full_str

  return full_str[len(prefix):]


def main(argv):
  src_dir = argv[1]
  dest_dir = argv[2]
  print("from %s to %s" % (src_dir, dest_dir))
  os.makedirs(dest_dir, exist_ok=True)
  for root, _, files in os.walk(src_dir):
    for f in files:
      src = os.path.join(root, f)
      dest = os.path.join(dest_dir, removeprefix(src, src_dir).lstrip("/"))

      maybe_create_dir(dest)

      _, ext = os.path.splitext(f)
      if ext != ".html":
        shutil.copyfile(src, dest)
      else:
        transform(src, dest)


if __name__ == "__main__":
  app.run(main)
