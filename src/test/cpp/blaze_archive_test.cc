// Copyright 2023 The Bazel Authors. All rights reserved.
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

#include "file/base/helpers.h"
#include "file/base/path.h"
#include "file/util/temp_path.h"
#include "file/zipfile/zipfilewriter.h"
#include "src/main/cpp/archive_utils.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "third_party/absl/strings/escaping.h"
#include "src/main/cpp/blaze.h"
#include "src/main/cpp/util/file_platform.h"

using ::testing::Gt;
using ::testing::status::IsOkAndHolds;

namespace blaze {

static absl::StatusOr<std::string> MakeZipAndReturnInstallBase(
    absl::string_view path, std::vector<std::pair<std::string, std::string>>
                                blaze_zip_file_to_contents) {
  ASSIGN_OR_RETURN(auto writer,
                   file_zipfile::ZipfileWriter::Create(path, file::Defaults()));
  for (const auto file_and_contents : blaze_zip_file_to_contents) {
    writer->AddFileFromString(file_and_contents.first,
                              file_and_contents.second);
  }
  RETURN_IF_ERROR(writer->CloseWithStatus(file::Defaults()));
  return "install_base";
}

auto get_mtime = [](absl::string_view path) -> absl::StatusOr<absl::Time> {
  ASSIGN_OR_RETURN(
      const auto stat,
      file::Stat(path, file::StatMask(tech::file::STAT_MTIME_NSECS)));
  return absl::FromUnixNanos(stat.mtime_nsecs());
};

// TODO(b/269617634) - add more tests to formalize archive unpacking.
class BlazeArchiveTest : public ::testing::Test {
 protected:
  BlazeArchiveTest() {}

  virtual ~BlazeArchiveTest() {}

  const TempPath temp_{TempPath::Local};
};

TEST_F(BlazeArchiveTest, TestZipExtractionAndFarOutMTimes) {
  const std::string blaze_path = file::JoinPath(temp_.path(), "blaze");
  std::vector<std::pair<std::string, std::string>> blaze_zip_file_to_contents;
  blaze_zip_file_to_contents.push_back(std::make_pair("foo", "foo content"));
  blaze_zip_file_to_contents.push_back(std::make_pair("bar", "bar content"));
  blaze_zip_file_to_contents.push_back(
      std::make_pair("path/to/subdir/baz", "baz content"));
  blaze_zip_file_to_contents.push_back(
      std::make_pair("install_base_key", "expected_install_md5"));
  ASSERT_OK_AND_ASSIGN(
      const std::string install_base,
      MakeZipAndReturnInstallBase(blaze_path, blaze_zip_file_to_contents));
  const std::string output_dir = file::JoinPath(temp_.path(), install_base);
  ASSERT_OK(file::RecursivelyCreateDir(output_dir, file::CreationMode(0750)));

  ExtractArchiveOrDie(blaze_path, "blaze", "expected_install_md5", output_dir);
  BlessFiles(output_dir);

  const std::string foo_path = file::JoinPath(output_dir, "foo");
  const std::string bar_path = file::JoinPath(output_dir, "bar");
  const std::string baz_path = file::JoinPath(output_dir, "path/to/subdir/baz");

  EXPECT_THAT(file::GetContents(foo_path, file::Defaults()),
              IsOkAndHolds("foo content"));
  EXPECT_THAT(file::GetContents(bar_path, file::Defaults()),
              IsOkAndHolds("bar content"));
  EXPECT_THAT(file::GetContents(baz_path, file::Defaults()),
              IsOkAndHolds("baz content"));

  std::unique_ptr<blaze_util::IFileMtime> mtime(blaze_util::CreateFileMtime());
  EXPECT_TRUE(mtime->IsUntampered(blaze_util::Path(foo_path)));
  EXPECT_TRUE(mtime->IsUntampered(blaze_util::Path(bar_path)));
  EXPECT_TRUE(mtime->IsUntampered(blaze_util::Path(baz_path)));

  const auto far_future = absl::Now() + absl::Hours(24 * 365 * 9);
  EXPECT_THAT(get_mtime(foo_path), IsOkAndHolds(Gt(far_future)));
  EXPECT_THAT(get_mtime(bar_path), IsOkAndHolds(Gt(far_future)));
  EXPECT_THAT(get_mtime(baz_path), IsOkAndHolds(Gt(far_future)));
}
}  // namespace blaze
