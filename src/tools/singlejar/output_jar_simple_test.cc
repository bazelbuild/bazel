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

#include "src/main/cpp/blaze_util.h"
#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"
#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/options.h"
#include "src/tools/singlejar/output_jar.h"
#include "gtest/gtest.h"

namespace {

#if !defined(DATA_DIR_TOP)
#define DATA_DIR_TOP
#endif

static bool HasSubstr(const std::string &s, const std::string &what) {
  return std::string::npos != s.find(what);
}

class OutputJarSimpleTest : public ::testing::Test {
 protected:
  OutputJarSimpleTest() {}

  ~OutputJarSimpleTest() override {}

  void CreateOutput(const std::string &out_path, const char *first_arg...) {
    std::string args_string;
    va_list ap;
    va_start(ap, first_arg);
    const char *args[100] = {"--output", out_path.c_str()};
    unsigned nargs = 2;
    if (first_arg) {
      args[nargs++] = first_arg;
      while (nargs < arraysize(args)) {
        const char *arg = va_arg(ap, const char *);
        if (arg) {
          args[nargs++] = arg;
          args_string += ' ';
          args_string += arg;
        } else {
          break;
        }
      }
      va_end(ap);
      ASSERT_GE(arraysize(args), nargs);
    }
    printf("Arguments: %s\n", args_string.c_str());
    options_.ParseCommandLine(nargs, args);
    ASSERT_EQ(0, output_jar_.Doit(&options_));
    EXPECT_EQ(0, VerifyZip(out_path));
  }

  static std::string OutputFilePath(const std::string &relpath) {
    const char *out_dir = getenv("TEST_TMPDIR");
    return blaze_util::JoinPath((nullptr == out_dir) ? "." : out_dir,
                                relpath.c_str());
  }

  static int VerifyZip(const std::string &zip_path) {
    std::string verify_command;
    blaze_util::StringPrintf(&verify_command, "zip -Tv %s", zip_path.c_str());
    return system(verify_command.c_str());
  }

  static void GetEntryContents(const std::string &zip_path,
                               const char *entry_name, std::string *contents) {
    contents->clear();
    std::string command;
    blaze_util::StringPrintf(&command, "unzip -p %s %s", zip_path.c_str(),
                             entry_name);
    FILE *fp = popen(command.c_str(), "r");
    ASSERT_NE(nullptr, fp);
    char buf[1024];
    while (fgets(buf, sizeof(buf), fp)) {
      contents->append(buf);
    }
    ASSERT_NE(0, feof(fp));
    ASSERT_EQ(0, ferror(fp));
    ASSERT_EQ(0, pclose(fp));
  }

  OutputJar output_jar_;
  Options options_;
};

// No inputs at all.
TEST_F(OutputJarSimpleTest, Empty) {
  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, nullptr);
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path.c_str()));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    ASSERT_TRUE(cdh->is()) << "No expected tag in the Central Directory Entry.";
    ASSERT_NE(nullptr, lh) << "No local header.";
    ASSERT_TRUE(lh->is()) << "No expected tag in the Local Header.";
    EXPECT_EQ(lh->file_name_string(), cdh->file_name_string());
    if (!cdh->no_size_in_local_header()) {
      EXPECT_EQ(lh->compressed_file_size(), cdh->compressed_file_size())
          << "Entry: " << lh->file_name_string();
      EXPECT_EQ(lh->uncompressed_file_size(), cdh->uncompressed_file_size())
          << "Entry: " << cdh->file_name_string();
    }
  }
  input_jar.Close();
  std::string manifest;
  GetEntryContents(out_path, "META-INF/MANIFEST.MF", &manifest);
  EXPECT_EQ(
      "Manifest-Version: 1.0\r\n"
      "Created-By: singlejar\r\n"
      "\r\n",
      manifest);
  std::string build_properties;
  GetEntryContents(out_path, "build-data.properties", &build_properties);
  EXPECT_PRED2(HasSubstr, build_properties, "build.target=");
}

// Source jars.
TEST_F(OutputJarSimpleTest, Source) {
  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, "--sources",
               DATA_DIR_TOP "src/tools/singlejar/libtest1.jar",
               DATA_DIR_TOP "src/tools/singlejar/libtest2.jar", nullptr);
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path.c_str()));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    ASSERT_TRUE(cdh->is()) << "No expected tag in the Central Directory Entry.";
    ASSERT_NE(nullptr, lh) << "No local header.";
    ASSERT_TRUE(lh->is()) << "No expected tag in the Local Header.";
    EXPECT_EQ(lh->file_name_string(), cdh->file_name_string());
    if (!cdh->no_size_in_local_header()) {
      EXPECT_EQ(lh->compressed_file_size(), cdh->compressed_file_size())
          << "Entry: " << lh->file_name_string();
      EXPECT_EQ(lh->uncompressed_file_size(), cdh->uncompressed_file_size())
          << "Entry: " << cdh->file_name_string();
    }
  }
  input_jar.Close();
}

// Verify --java_launcher argument
TEST_F(OutputJarSimpleTest, JavaLauncher) {
  std::string out_path = OutputFilePath("out.jar");
  const char *launcher_path = DATA_DIR_TOP "src/tools/singlejar/libtest1.jar";
  CreateOutput(out_path, "--java_launcher", launcher_path, nullptr);
  // check that the offset of the first entry equals launcher size.
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path.c_str()));
  const LH *lh;
  const CDH *cdh;
  cdh = input_jar.NextEntry(&lh);
  ASSERT_NE(nullptr, cdh);
  struct stat statbuf;
  ASSERT_EQ(0, stat(launcher_path, &statbuf));
  EXPECT_TRUE(cdh->is());
  EXPECT_TRUE(lh->is());
  EXPECT_EQ(statbuf.st_size, cdh->local_header_offset());
  input_jar.Close();
}

// --main_class option.
TEST_F(OutputJarSimpleTest, MainClass) {
  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, "--main_class", "com.google.my.Main", nullptr);
  std::string manifest;
  GetEntryContents(out_path, "META-INF/MANIFEST.MF", &manifest);
  EXPECT_EQ(
      "Manifest-Version: 1.0\r\n"
      "Created-By: singlejar\r\n"
      "Main-Class: com.google.my.Main\r\n"
      "\r\n",
      manifest);
}

// --deploy_manifest_lines option.
TEST_F(OutputJarSimpleTest, DeployManifestLines) {
  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, "--deploy_manifest_lines", "property1: foo",
               "property2: bar", nullptr);
  std::string manifest;
  GetEntryContents(out_path, "META-INF/MANIFEST.MF", &manifest);
  EXPECT_EQ(
      "Manifest-Version: 1.0\r\n"
      "Created-By: singlejar\r\n"
      "property1: foo\r\n"
      "property2: bar\r\n"
      "\r\n",
      manifest);
}

// --extra_build_info option
TEST_F(OutputJarSimpleTest, ExtraBuildInfo) {
  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, "--extra_build_info", "property1=value1",
               "--extra_build_info", "property2=value2", nullptr);
  std::string build_properties;
  GetEntryContents(out_path, "build-data.properties", &build_properties);
  EXPECT_PRED2(HasSubstr, build_properties, "\nproperty1=value1\n");
  EXPECT_PRED2(HasSubstr, build_properties, "\nproperty2=value2\n");
}

// --build_info_file and --extra_build_info options.
TEST_F(OutputJarSimpleTest, BuildInfoFile) {
  std::string build_info_path1 = OutputFilePath("buildinfo1");
  ASSERT_TRUE(blaze::WriteFile("property11=value11\nproperty12=value12\n",
                               build_info_path1));
  std::string build_info_path2 = OutputFilePath("buildinfo2");
  ASSERT_TRUE(blaze::WriteFile("property21=value21\nproperty22=value22\n",
                               build_info_path2));
  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, "--build_info_file", build_info_path1.c_str(),
               "--extra_build_info", "property=value", "--build_info_file",
               build_info_path2.c_str(), nullptr);
  std::string build_properties;
  GetEntryContents(out_path, "build-data.properties", &build_properties);
  EXPECT_PRED2(HasSubstr, build_properties, "property11=value11\n");
  EXPECT_PRED2(HasSubstr, build_properties, "property12=value12\n");
  EXPECT_PRED2(HasSubstr, build_properties, "property21=value21\n");
  EXPECT_PRED2(HasSubstr, build_properties, "property22=value22\n");
  EXPECT_PRED2(HasSubstr, build_properties, "property=value\n");
}

// --resources option.
TEST_F(OutputJarSimpleTest, Resources) {
  std::string res11_path = OutputFilePath("res11");
  std::string res11_spec = std::string("res1:") + res11_path;
  ASSERT_TRUE(blaze::WriteFile("res11.line1\nres11.line2\n", res11_path));

  std::string res12_path = OutputFilePath("res12");
  std::string res12_spec = std::string("res1:") + res12_path;
  ASSERT_TRUE(blaze::WriteFile("res12.line1\nres12.line2\n", res12_path));

  std::string res2_path = OutputFilePath("res2");
  ASSERT_TRUE(blaze::WriteFile("res2.line1\nres2.line2\n", res2_path));

  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, "--resources", res11_spec.c_str(), res12_spec.c_str(),
               res2_path.c_str(), nullptr);

  // The output should have 'res1' entry containing the concatenation of the
  // 'res11' and 'res12' files.
  std::string res1;
  GetEntryContents(out_path, "res1", &res1);
  EXPECT_EQ("res11.line1\nres11.line2\nres12.line1\nres12.line2\n", res1);

  // The output should have res2 path entry and contents.
  std::string res2;
  GetEntryContents(out_path, res2_path.c_str(), &res2);
  EXPECT_EQ("res2.line1\nres2.line2\n", res2);
}

// --classpath_resources
TEST_F(OutputJarSimpleTest, ClasspathResources) {
  std::string res1_path = OutputFilePath("cp_res");
  ASSERT_TRUE(blaze::WriteFile("line1\nline2\n", res1_path));
  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, "--classpath_resources", res1_path.c_str(), nullptr);
  std::string res;
  GetEntryContents(out_path, "cp_res", &res);
  EXPECT_EQ("line1\nline2\n", res);
}

// Duplicate entries for --resources or --classpath_resources
TEST_F(OutputJarSimpleTest, DuplicateResources) {
  std::string cp_res_path = OutputFilePath("cp_res");
  ASSERT_TRUE(blaze::WriteFile("line1\nline2\n", cp_res_path));

  std::string res1_path = OutputFilePath("res1");
  std::string res1_spec = "foo:" + res1_path;
  ASSERT_TRUE(blaze::WriteFile("resline1\nresline2\n", res1_path));

  std::string res2_path = OutputFilePath("res2");
  std::string res2_spec = "foo:" + res2_path;
  ASSERT_TRUE(blaze::WriteFile("line3\nline4\n", res2_path));

  std::string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, "--warn_duplicate_resources", "--resources",
               res1_spec.c_str(), res2_spec.c_str(), "--classpath_resources",
               cp_res_path.c_str(), cp_res_path.c_str(), nullptr);

  std::string cp_res;
  GetEntryContents(out_path, "cp_res", &cp_res);
  EXPECT_EQ("line1\nline2\n", cp_res);

  std::string foo;
  GetEntryContents(out_path, "foo", &foo);
  EXPECT_EQ("resline1\nresline2\n", foo);
}

}  // namespace
