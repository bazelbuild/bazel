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
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"

class BazelOutputServiceImpl
    : public bazel_output_service::BazelOutputService::Service {
  grpc::Status Clean(grpc::ServerContext* context,
                     const bazel_output_service::CleanRequest* request,
                     bazel_output_service::CleanResponse* response) override;

  grpc::Status StartBuild(
      grpc::ServerContext* context,
      const bazel_output_service::StartBuildRequest* request,
      bazel_output_service::StartBuildResponse* response) override;

  grpc::Status StageArtifacts(
      grpc::ServerContext* context,
      const bazel_output_service::StageArtifactsRequest* request,
      bazel_output_service::StageArtifactsResponse* response) override;

  grpc::Status FinalizeArtifacts(
      grpc::ServerContext* context,
      const bazel_output_service::FinalizeArtifactsRequest* request,
      bazel_output_service::FinalizeArtifactsResponse* response) override;

  grpc::Status FinalizeBuild(
      grpc::ServerContext* context,
      const bazel_output_service::FinalizeBuildRequest* request,
      bazel_output_service::FinalizeBuildResponse* response) override;

  grpc::Status BatchStat(
      grpc::ServerContext* context,
      const bazel_output_service::BatchStatRequest* request,
      bazel_output_service::BatchStatResponse* response) override;
};

int RunServer(int argc, char** argv);

#endif  // BAZEL_SRC_TOOLS_REMOTE_SRC_MAIN_CPP_OUTPUT_SERVICE_BAZEL_OUTPUT_SERVICE_IMPL_H_
