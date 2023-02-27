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
#include "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library/bar.h"
#include "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library/baz.h"
#include "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library/direct_so_file_cc_lib.h"
#include "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library/direct_so_file_cc_lib2.h"
#include "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library/qux.h"
#include "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library3/diff_pkg.h"

int foo() {
  diff_pkg();
  bar();
  baz();
  qux();
#ifdef IS_LINUX
  direct_so_file_cc_lib();
  direct_so_file_cc_lib2();
#endif
  return 42;
}
