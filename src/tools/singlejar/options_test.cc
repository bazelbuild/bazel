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

#include <memory>

#include "src/tools/singlejar/options.h"

#include "gtest/gtest.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

class OptionsTest : public testing::Test {
 protected:
  void SetUp() override { options_.reset(new Options); }
  std::unique_ptr<Options> options_;
};

TEST_F(OptionsTest, Flags1) {
  const char *args[] = {"--exclude_build_data",
                        "--compression",
                        "--normalize",
                        "--no_duplicates",
                        "--output", "output_jar"};
  options_->ParseCommandLine(ARRAY_SIZE(args), args);

  EXPECT_TRUE(options_->exclude_build_data);
  EXPECT_TRUE(options_->force_compression);
  EXPECT_TRUE(options_->normalize_timestamps);
  EXPECT_TRUE(options_->no_duplicates);
  EXPECT_FALSE(options_->preserve_compression);
  EXPECT_FALSE(options_->verbose);
  EXPECT_FALSE(options_->warn_duplicate_resources);
  EXPECT_EQ("output_jar", options_->output_jar);
}

TEST_F(OptionsTest, Flags2) {
  const char *args[] = {"--dont_change_compression",
                        "--verbose",
                        "--warn_duplicate_resources",
                        "--output", "output_jar"};
  options_->ParseCommandLine(ARRAY_SIZE(args), args);

  ASSERT_FALSE(options_->exclude_build_data);
  ASSERT_FALSE(options_->force_compression);
  ASSERT_FALSE(options_->normalize_timestamps);
  ASSERT_FALSE(options_->no_duplicates);
  ASSERT_TRUE(options_->preserve_compression);
  ASSERT_TRUE(options_->verbose);
  ASSERT_TRUE(options_->warn_duplicate_resources);
}

TEST_F(OptionsTest, SingleOptargs) {
  const char *args[] = {"--output", "output_jar",
                        "--main_class", "com.google.Main",
                        "--java_launcher", "//tools:mylauncher",
                        "--build_info_file", "build_file1",
                        "--extra_build_info", "extra_build_line1",
                        "--build_info_file", "build_file2",
                        "--extra_build_info", "extra_build_line2"};
  options_->ParseCommandLine(ARRAY_SIZE(args), args);

  EXPECT_EQ("output_jar", options_->output_jar);
  EXPECT_EQ("com.google.Main", options_->main_class);
  EXPECT_EQ("//tools:mylauncher", options_->java_launcher);
  ASSERT_EQ(2, options_->build_info_files.size());
  EXPECT_EQ("build_file1", options_->build_info_files[0]);
  EXPECT_EQ("build_file2", options_->build_info_files[1]);
  ASSERT_EQ(2, options_->build_info_lines.size());
  EXPECT_EQ("extra_build_line1", options_->build_info_lines[0]);
  EXPECT_EQ("extra_build_line2", options_->build_info_lines[1]);
}

TEST_F(OptionsTest, MultiOptargs) {
  const char *args[] = {"--output", "output_file",
                        "--sources", "jar1", "jar2",
                        "--resources", "res1", "res2",
                        "--classpath_resources", "cpres1", "cpres2",
                        "--sources", "jar3",
                        "--include_prefixes", "prefix1", "prefix2"};
  options_->ParseCommandLine(ARRAY_SIZE(args), args);

  ASSERT_EQ(3, options_->input_jars.size());
  EXPECT_EQ("jar1", options_->input_jars[0]);
  EXPECT_EQ("jar2", options_->input_jars[1]);
  EXPECT_EQ("jar3", options_->input_jars[2]);
  ASSERT_EQ(2, options_->resources.size());
  EXPECT_EQ("res1", options_->resources[0]);
  EXPECT_EQ("res2", options_->resources[1]);
  ASSERT_EQ(2, options_->classpath_resources.size());
  EXPECT_EQ("cpres1", options_->classpath_resources[0]);
  EXPECT_EQ("cpres2", options_->classpath_resources[1]);
  ASSERT_EQ(2, options_->include_prefixes.size());
  EXPECT_EQ("prefix1", options_->include_prefixes[0]);
  EXPECT_EQ("prefix2", options_->include_prefixes[1]);
}

TEST_F(OptionsTest, EmptyMultiOptargs) {
  const char *args[] = {"--output", "output_file",
                        "--sources",
                        "--resources",
                        "--classpath_resources",
                        "--sources",
                        "--include_prefixes", "prefix1",
                        "--resources"};
  options_->ParseCommandLine(ARRAY_SIZE(args), args);

  EXPECT_EQ(0, options_->input_jars.size());
  EXPECT_EQ(0, options_->resources.size());
  EXPECT_EQ(0, options_->classpath_resources.size());
  EXPECT_EQ(1, options_->include_prefixes.size());
}
