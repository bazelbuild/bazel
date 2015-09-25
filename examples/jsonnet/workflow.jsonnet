// Copyright 2015 The Bazel Authors. All rights reserved.
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

// Configuration for a hypothetical workflow scheduling system.
{
  // Configuration for a workflow.
  Workflow:: {
    schedule: {},
    retries: 5,
    jobs: {},
  },

  // Scheduling configuration for a workflow.
  Schedule:: {
    start_date: "",
    start_time: "",
    repeat_frequency: 0,
    repeat_type: "",
  },

  // Base configuration for a Job in a workflow.
  Job:: {
    type: "base",
    deps: [],
    inputs: [],
    outputs: [],
  },

  // Configuration for a job that runs a shell command.
  ShJob:: self.Job {
    type: "sh",
    command: "",
    vars: {},
  }
}
