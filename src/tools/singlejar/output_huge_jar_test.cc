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

#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"
#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/options.h"
#include "src/tools/singlejar/output_jar.h"
#include "src/tools/singlejar/test_util.h"
#include "src/tools/singlejar/zip_headers.h"
#include "googletest/include/gtest/gtest.h"

namespace {

using rules_cc::cc::runfiles::Runfiles;
using singlejar_test_util::AllocateFile;
using singlejar_test_util::OutputFilePath;
using singlejar_test_util::VerifyZip;

using std::string;

class OutputHugeJarTest : public ::testing::Test {
 protected:
  void SetUp() override { runfiles.reset(Runfiles::CreateForTest()); }

  void CreateOutput(const string& out_path, const std::vector<string>& args) {
    const char* option_list[100] = {"--output", out_path.c_str()};
    int nargs = 2;
    for (auto& arg : args) {
      if (arg.empty()) {
        continue;
      }
      option_list[nargs++] = arg.c_str();
      if (arg.find(' ') == string::npos) {
        fprintf(stderr, " '%s'", arg.c_str());
      } else {
        fprintf(stderr, " %s", arg.c_str());
      }
    }
    fprintf(stderr, "\n");
    options_.ParseCommandLine(nargs, option_list);
    OutputJar output_jar_(&options_);
    ASSERT_EQ(0, output_jar_.Doit());
    EXPECT_EQ(0, VerifyZip(out_path));
  }

  Options options_;
  std::unique_ptr<Runfiles> runfiles;
};

TEST_F(OutputHugeJarTest, EntryAbove4G) {
  // Verifies that an entry above 4G is handled correctly.

  // Have huge launcher, then the first jar entry will be above 4G.
  string launcher_path = OutputFilePath("launcher");
  ASSERT_TRUE(AllocateFile(launcher_path, 0x100000010));

  string out_path = OutputFilePath("out.jar");
  CreateOutput(
      out_path,
      {"--java_launcher", launcher_path, "--sources",
       runfiles
           ->Rlocation(
               "io_bazel/src/tools/singlejar/libtest1.jar")
           .c_str()});
}

TEST_F(OutputHugeJarTest, ExtraFieldsOverflowAbove4G) {
  string launcher_path = OutputFilePath("launcher_ef");
  ASSERT_TRUE(AllocateFile(launcher_path, 0x100000010));

  std::string zip_data;
  const std::string filename = "entry.txt";

  size_t lh_offset = zip_data.size();
  size_t lh_size = sizeof(LH) + filename.size();
  zip_data.resize(lh_offset + lh_size, 0);
  auto* lh = reinterpret_cast<LH*>(&zip_data[lh_offset]);
  lh->signature();
  lh->version(20);
  lh->file_name(filename.data(), filename.size());

  // Non-Zip64 extra field of size 0xFFF8 (4-byte header + 0xFFF4 payload).
  // Adding a 12-byte Zip64 extra field when placed above 4G exceeds 0xFFFF.
  std::vector<uint8_t> ef_buffer(0xFFF8, 0);
  auto* ef = reinterpret_cast<ExtraField*>(ef_buffer.data());
  ef->signature(0xCAFE);
  ef->payload_size(0xFFF4);

  size_t cdh_offset = zip_data.size();
  size_t cdh_size = sizeof(CDH) + filename.size() + ef_buffer.size();
  zip_data.resize(cdh_offset + cdh_size, 0);
  auto* cdh = reinterpret_cast<CDH*>(&zip_data[cdh_offset]);
  cdh->signature();
  cdh->version(20);
  cdh->version_to_extract(20);
  cdh->local_header_offset32(lh_offset);
  cdh->file_name(filename.data(), filename.size());
  cdh->extra_fields(ef_buffer.data(), ef_buffer.size());

  size_t ecd_offset = zip_data.size();
  zip_data.resize(ecd_offset + sizeof(ECD), 0);
  auto* ecd = reinterpret_cast<ECD*>(&zip_data[ecd_offset]);
  ecd->signature();
  ecd->this_disk_entries16(1);
  ecd->total_entries16(1);
  ecd->cen_size32(cdh_size);
  ecd->cen_offset32(cdh_offset);

  string in_jar = OutputFilePath("large_ef.jar");
  string out_path = OutputFilePath("out_ef.jar");
  ASSERT_TRUE(blaze_util::WriteFile(zip_data, in_jar));

  const char* option_list[] = {"--output",        out_path.c_str(),
                               "--java_launcher", launcher_path.c_str(),
                               "--sources",       in_jar.c_str()};
  options_.ParseCommandLine(6, option_list);
  OutputJar output_jar(&options_);
  EXPECT_DEATH(output_jar.Doit(), "extra fields size .* exceeds 64KB");
}

}  // namespace
