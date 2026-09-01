// Copyright 2026 The Bazel Authors. All rights reserved.
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

namespace bazel {
namespace tools {
namespace test_wrapper {

int TestWrapperMain(int argc, char** argv);

}  // namespace test_wrapper
}  // namespace tools
}  // namespace bazel

int main(int argc, char** argv) {
  return bazel::tools::test_wrapper::TestWrapperMain(argc, argv);
}
