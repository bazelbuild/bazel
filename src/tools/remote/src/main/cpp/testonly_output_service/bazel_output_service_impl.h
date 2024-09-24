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

#ifndef BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_OUTPUT_SERVICE_BAZEL_OUTPUT_SERVICE_IMPL_H_
#define BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_OUTPUT_SERVICE_BAZEL_OUTPUT_SERVICE_IMPL_H_

#include "src/main/protobuf/bazel_output_service.grpc.pb.h"

class BazelOutputServiceImpl
    : public bazel_output_service::BazelOutputService::Service {};

#endif  // BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_OUTPUT_SERVICE_BAZEL_OUTPUT_SERVICE_IMPL_H_
