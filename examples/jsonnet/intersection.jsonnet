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

local workflow = import "examples/jsonnet/workflow.jsonnet";

// Workflow that performs an intersection of two files using shell commands.
{
  intersection: workflow.Workflow {
    jobs: {
      local input_file1 = "/tmp/list1",
      local input_file2 = "/tmp/list2",
      local sorted_file1 = "/tmp/list1_sorted",
      local sorted_file2 = "/tmp/list2_sorted",
      local intersection = "/tmp/intersection",

      SortJob:: workflow.ShJob {
        input_file:: "",
        output_file:: "",
        command: "sort %s > %s" % [self.input_file, self.output_file],
        inputs: [self.input_file],
        outputs: [self.output_file],
      },

      sort_file1: self.SortJob {
        input_file:: input_file1,
        output_file:: sorted_file1,
      },

      sort_file2: self.SortJob {
        input_file:: input_file2,
        output_file:: sorted_file2,
      },

      intersect: workflow.ShJob {
        deps: [
          ":sort_file1",
          ":sort_file2",
        ],
        command: "comm -12 %s %s > %s" %
            [sorted_file1, sorted_file2, intersection],
        inputs: [
          sorted_file1,
          sorted_file2,
        ],
        outputs: [intersection],
      },
    }
  }
}
