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

#ifndef THIRD_PARTY_BAZEL_SRC_MAIN_CPP_STARTUP_INTERCEPTOR_H_
#define THIRD_PARTY_BAZEL_SRC_MAIN_CPP_STARTUP_INTERCEPTOR_H_

#include <string_view>

#include "src/main/cpp/startup_options.h"

namespace blaze {

// An interface for rerouting an invocation to a different backend
class StartupInterceptor {
 public:
  StartupInterceptor() = default;

  virtual ~StartupInterceptor() = default;

  // Reroutes the invocation based on startup options, build label, and other
  // criteria.
  virtual void MaybeReroute(StartupOptions const* startup_options,
                            std::string_view build_label) = 0;
};

}  // namespace blaze

#endif  // THIRD_PARTY_BAZEL_SRC_MAIN_CPP_STARTUP_INTERCEPTOR_H_
