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

#ifndef BAZEL_TOOLS_CPP_MODULE_TOOLS_AGGREGATE_DDI_H_
#define BAZEL_TOOLS_CPP_MODULE_TOOLS_AGGREGATE_DDI_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "common/common.h"

void write_output(std::ostream &output, const Cpp20ModulesInfo &info);

#endif  // BAZEL_TOOLS_CPP_MODULE_TOOLS_AGGREGATE_DDI_H_
