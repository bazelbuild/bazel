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
#include <stdio.h>

// Must be included before anything else.
#include "src/tools/singlejar/port.h"

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/port.h"
#include "src/main/cpp/util/strings.h"
#include "src/tools/singlejar/input_jar.h"
#include "src/tools/singlejar/options.h"
#include "src/tools/singlejar/output_jar.h"
#include "src/tools/singlejar/test_util.h"
#include "googletest/include/gtest/gtest.h"

#ifdef _WIN32
const char JAR_TOOL_PATH[] = "local_jdk/bin/jar.exe";
#define unlink _unlink
#else
const char JAR_TOOL_PATH[] = "local_jdk/bin/jar";
#endif

namespace {

using bazel::tools::cpp::runfiles::Runfiles;
using singlejar_test_util::CreateTextFile;
using singlejar_test_util::GetEntryContents;
using singlejar_test_util::GetEntryContents;
using singlejar_test_util::OutputFilePath;
using singlejar_test_util::RunCommand;
using singlejar_test_util::VerifyZip;

using std::string;

const char kPathLibData1[] = "io_bazel/src/tools/singlejar/libdata1.jar";
const char kPathLibData2[] = "io_bazel/src/tools/singlejar/libdata2.jar";

static bool HasSubstr(const string &s, const string &what) {
  return string::npos != s.find(what);
}

// A subclass of the OutputJar which concatenates the contents of each
// entry in the data/ directory from the input archives.
class CustomOutputJar : public OutputJar {
 public:
  ~CustomOutputJar() override {}
  void ExtraHandler(const CDH *cdh,
                    const std::string *input_jar_aux_label) override {
    auto file_name = cdh->file_name();
    auto file_name_length = cdh->file_name_length();
    if (file_name_length > 0 && file_name[file_name_length - 1] != '/' &&
        begins_with(file_name, file_name_length, "tools/singlejar/data/")) {
      // The contents of the data/<FILE> on the output is the
      // concatenation of the data/<FILE> files from all inputs.
      std::string metadata_file_path(file_name, file_name_length);
      if (NewEntry(metadata_file_path)) {
        ExtraCombiner(metadata_file_path, new Concatenator(metadata_file_path));
      }
    }
  }
};

class OutputJarSimpleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    runfiles.reset(Runfiles::CreateForTest());
  }

  void CreateOutput(const string &out_path, const std::vector<string> &args) {
    const char *option_list[100] = {"--output", out_path.c_str()};
    int nargs = 2;
    for (auto &arg : args) {
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
    ASSERT_EQ(0, output_jar_.Doit(&options_));
    EXPECT_EQ(0, VerifyZip(out_path));
  }

  string CompressionOptionsTestingJar(const string &compression_option) {
    string cp_res_path =
        CreateTextFile("cp_res", "line1\nline2\nline3\nline4\n");
    string out_path = OutputFilePath("out.jar");
    CreateOutput(
        out_path,
        {compression_option, "--sources",
         runfiles->Rlocation("io_bazel/src/tools/singlejar/libtest1.jar")
             .c_str(),
         runfiles->Rlocation("io_bazel/src/tools/singlejar/stored.jar").c_str(),
         "--resources", cp_res_path, "--deploy_manifest_lines",
         "property1: value1", "property2: value2"});
    return out_path;
  }

  OutputJar output_jar_;
  Options options_;
  std::unique_ptr<Runfiles> runfiles;
};

// No inputs at all.
TEST_F(OutputJarSimpleTest, Empty) {
  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {});
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  int entry_count = 0;
  const LH *lh;
  const CDH *cdh;
  const uint8_t cafe_extra_field[] = {0xFE, 0xCA, 0, 0};
  while ((cdh = input_jar.NextEntry(&lh))) {
    ++entry_count;
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
    // Verify that each entry has a reasonable timestamp.
    EXPECT_EQ(lh->last_mod_file_date(), cdh->last_mod_file_date())
        << "Entry: " << lh->file_name_string();
    EXPECT_EQ(lh->last_mod_file_time(), cdh->last_mod_file_time())
        << "Entry: " << lh->file_name_string();
    uint16_t dos_time = lh->last_mod_file_time();
    uint16_t dos_date = lh->last_mod_file_date();

    // Current time, rounded to even number of seconds because MSDOS timestamp
    // does this, too.
    time_t now = (time(nullptr) + 1) & ~1;
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char now_time_str[50];
    strftime(now_time_str, sizeof(now_time_str), "%c", &tm_now);

    // Unpack MSDOS file timestamp. See the comment about its format in
    // output_jar.cc.
    struct tm tm;
    tm.tm_sec = (dos_time & 31) << 1;
    tm.tm_min = (dos_time >> 5) & 63;
    tm.tm_hour = (dos_time >> 11) & 31;
    tm.tm_mday = (dos_date & 31);
    tm.tm_mon = ((dos_date >> 5) & 15) - 1;
    tm.tm_year = ((dos_date >> 9) & 127) + 80;
    tm.tm_isdst = tm_now.tm_isdst;
    time_t entry_time = mktime(&tm);
    char entry_time_str[50];
    strftime(entry_time_str, sizeof(entry_time_str), "%c", &tm);

    // Without --normalize option all the entries should have reasonably
    // current timestamp (which we arbitrarily choose to be <5 minutes).
    EXPECT_GE(now, entry_time) << now_time_str << " vs. " << entry_time_str;
    EXPECT_LE(now, entry_time + 300) << now_time_str << " vs. "
                                     << entry_time_str;

    // The first entry should be for the META-INF/ directory, and it should
    // contain a single extra field 0xCAFE. Although
    // https://bugs.openjdk.java.net/browse/JDK-6808540 claims that this extra
    // field is optional, 'file' utility in Linux relies on to distinguish
    // jar from zip.
    if (entry_count == 1) {
      ASSERT_EQ("META-INF/", lh->file_name_string());
      ASSERT_EQ(4, lh->extra_fields_length());
      ASSERT_EQ(0, memcmp(cafe_extra_field, lh->extra_fields(), 4));
      ASSERT_EQ(4, cdh->extra_fields_length());
      ASSERT_EQ(0, memcmp(cafe_extra_field, cdh->extra_fields(), 4));
    }
  }
  input_jar.Close();
  string manifest = GetEntryContents(out_path, "META-INF/MANIFEST.MF");
  EXPECT_EQ(
      "Manifest-Version: 1.0\r\n"
      "Created-By: singlejar\r\n"
      "\r\n",
      manifest);
  string build_properties = GetEntryContents(out_path, "build-data.properties");
  EXPECT_PRED2(HasSubstr, build_properties, "build.target=");
}

// Source jars.
TEST_F(OutputJarSimpleTest, Source) {
  string out_path = OutputFilePath("out.jar");
  CreateOutput(
      out_path,
      {"--sources",
       runfiles->Rlocation("io_bazel/src/tools/singlejar/libtest1.jar").c_str(),
       runfiles->Rlocation("io_bazel/src/tools/singlejar/libtest2.jar")
           .c_str()});
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  int file_count = 0;
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
    if (lh->file_name()[lh->file_name_length() - 1] != '/') {
      ++file_count;
    }
  }
  ASSERT_LE(4, file_count);
  input_jar.Close();
}

// Verify --java_launcher argument
TEST_F(OutputJarSimpleTest, JavaLauncher) {
  string out_path = OutputFilePath("out.jar");
  std::string launcher_path =
      runfiles->Rlocation("io_bazel/src/tools/singlejar/libtest1.jar");
  CreateOutput(out_path, {"--java_launcher", launcher_path});
  // check that the offset of the first entry equals launcher size.
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path.c_str()));
  const LH *lh;
  const CDH *cdh;
  cdh = input_jar.NextEntry(&lh);
  ASSERT_NE(nullptr, cdh);
  struct stat statbuf;
  ASSERT_EQ(0, stat(launcher_path.c_str(), &statbuf));
  EXPECT_TRUE(cdh->is());
  EXPECT_TRUE(lh->is());
  EXPECT_EQ(statbuf.st_size, cdh->local_header_offset());
  input_jar.Close();
}

// --main_class option.
TEST_F(OutputJarSimpleTest, MainClass) {
  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--main_class", "com.google.my.Main"});
  string manifest = GetEntryContents(out_path, "META-INF/MANIFEST.MF");
  EXPECT_EQ(
      "Manifest-Version: 1.0\r\n"
      "Created-By: singlejar\r\n"
      "Main-Class: com.google.my.Main\r\n"
      "\r\n",
      manifest);
}

// --deploy_manifest_lines option.
TEST_F(OutputJarSimpleTest, DeployManifestLines) {
  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path,
               {"--deploy_manifest_lines", "property1: foo", "property2: bar"});
  string manifest = GetEntryContents(out_path, "META-INF/MANIFEST.MF");
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
  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--extra_build_info", "property1=value1",
                          "--extra_build_info", "property2=value2"});
  string build_properties = GetEntryContents(out_path, "build-data.properties");
  EXPECT_PRED2(HasSubstr, build_properties, "\nproperty1=value1\n");
  EXPECT_PRED2(HasSubstr, build_properties, "\nproperty2=value2\n");
}

// --build_info_file and --extra_build_info options.
TEST_F(OutputJarSimpleTest, BuildInfoFile) {
  string build_info_path1 =
      CreateTextFile("buildinfo1", "property11=value11\nproperty12=value12\n");
  string build_info_path2 =
      CreateTextFile("buildinfo2", "property21=value21\nproperty22=value22\n");

  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--build_info_file", build_info_path1,
                          "--extra_build_info", "property=value",
                          "--build_info_file", build_info_path2.c_str()});
  string build_properties = GetEntryContents(out_path, "build-data.properties");
  EXPECT_PRED2(HasSubstr, build_properties, "property11=value11\n");
  EXPECT_PRED2(HasSubstr, build_properties, "property12=value12\n");
  EXPECT_PRED2(HasSubstr, build_properties, "property21=value21\n");
  EXPECT_PRED2(HasSubstr, build_properties, "property22=value22\n");
  EXPECT_PRED2(HasSubstr, build_properties, "property=value\n");
}

// --resources option.
TEST_F(OutputJarSimpleTest, Resources) {
  string res11_path = CreateTextFile("res11", "res11.line1\nres11.line2\n");
  string res11_spec = res11_path + ":res1";

  string res12_path = CreateTextFile("res12", "res12.line1\nres12.line2\n");
  string res12_spec = res12_path + ":res1";

  string res2_path = CreateTextFile("res2", "res2.line1\nres2.line2\n");

  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--resources", res11_spec, res12_spec, res2_path});

  // The output should have 'res1' entry containing the concatenation of the
  // 'res11' and 'res12' files.
  string res1 = GetEntryContents(out_path, "res1");
  EXPECT_EQ("res11.line1\nres11.line2\nres12.line1\nres12.line2\n", res1);

  // The output should have res2 path entry and contents.
  string res2 = GetEntryContents(out_path, res2_path);
  EXPECT_EQ("res2.line1\nres2.line2\n", res2);
}

TEST_F(OutputJarSimpleTest, ResourcesParentDirectories) {
  string res1_path = CreateTextFile("res1", "res1.line1\nres1.line2\n");
  string res2_path = CreateTextFile("res2", "res2.line1\nres2.line2\n");

  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--exclude_build_data", "--resources",
                          res1_path + ":the/resources/res1",
                          res2_path + ":the/resources2/res2"});

  string res1 = GetEntryContents(out_path, "the/resources/res1");
  EXPECT_EQ("res1.line1\nres1.line2\n", res1);

  string res2 = GetEntryContents(out_path, "the/resources2/res2");
  EXPECT_EQ("res2.line1\nres2.line2\n", res2);

  // The output should contain entries for parent directories
  std::vector<string> expected_entries(
      {"META-INF/", "META-INF/MANIFEST.MF", "the/", "the/resources/",
       "the/resources/res1", "the/resources2/", "the/resources2/res2"});
  std::vector<string> jar_entries;
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    jar_entries.push_back(cdh->file_name_string());
  }
  input_jar.Close();
  EXPECT_EQ(expected_entries, jar_entries);
}

TEST_F(OutputJarSimpleTest, ResourcesDirectories) {
  string dir_path = OutputFilePath("resource_dir");
  blaze_util::MakeDirectories(dir_path, 0777);

  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path,
               {"--exclude_build_data", "--resources", dir_path + ":the/dir"});

  // The output should contain entries for the directory
  std::vector<string> expected_entries({
      "META-INF/", "META-INF/MANIFEST.MF", "the/", "the/dir/",
  });
  std::vector<string> jar_entries;
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    jar_entries.push_back(cdh->file_name_string());
  }
  input_jar.Close();
  EXPECT_EQ(expected_entries, jar_entries);
}

// --classpath_resources
TEST_F(OutputJarSimpleTest, ClasspathResources) {
  string res1_path = OutputFilePath("cp_res");
  ASSERT_TRUE(blaze_util::WriteFile("line1\nline2\n", res1_path));
  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--classpath_resources", res1_path.c_str()});
  string res = GetEntryContents(out_path, "cp_res");
  EXPECT_EQ("line1\nline2\n", res);
}

// Duplicate entries for --resources or --classpath_resources
TEST_F(OutputJarSimpleTest, DuplicateResources) {
  string cp_res_path = CreateTextFile("cp_res", "line1\nline2\n");

  string res1_path = CreateTextFile("res1", "resline1\nresline2\n");
  string res1_spec = res1_path + ":foo";

  string res2_path = CreateTextFile("res2", "line3\nline4\n");
  string res2_spec = res2_path + ":foo";

  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path,
               {"--warn_duplicate_resources", "--resources", res1_spec,
                res2_spec, "--classpath_resources", cp_res_path, cp_res_path});

  string cp_res = GetEntryContents(out_path, "cp_res");
  EXPECT_EQ("line1\nline2\n", cp_res);

  string foo = GetEntryContents(out_path, "foo");
  EXPECT_EQ("resline1\nresline2\n", foo);
}

// Extra combiners
TEST_F(OutputJarSimpleTest, ExtraCombiners) {
  string out_path = OutputFilePath("out.jar");
  const char kEntry[] = "tools/singlejar/data/extra_file1";
  output_jar_.ExtraCombiner(kEntry, new Concatenator(kEntry));
  CreateOutput(out_path, {"--sources", kPathLibData1, kPathLibData2});
  string contents1 = GetEntryContents(kPathLibData1, kEntry);
  string contents2 = GetEntryContents(kPathLibData2, kEntry);
  EXPECT_EQ(contents1 + contents2, GetEntryContents(out_path, kEntry));
}

// Test ExtraHandler override.
TEST_F(OutputJarSimpleTest, ExtraHandler) {
  string out_path = OutputFilePath("out.jar");
  const char kEntry[] = "tools/singlejar/data/extra_file1";
  const char *option_list[] = {"--output", out_path.c_str(), "--sources",
                               kPathLibData1, kPathLibData2};
  CustomOutputJar custom_output_jar;
  options_.ParseCommandLine(arraysize(option_list), option_list);
  ASSERT_EQ(0, custom_output_jar.Doit(&options_));
  EXPECT_EQ(0, VerifyZip(out_path));

  string contents1 = GetEntryContents(kPathLibData1, kEntry);
  string contents2 = GetEntryContents(kPathLibData2, kEntry);
  EXPECT_EQ(contents1 + contents2, GetEntryContents(out_path, kEntry));
}

// --include_headers
TEST_F(OutputJarSimpleTest, IncludeHeaders) {
  string out_path = OutputFilePath("out.jar");
  CreateOutput(
      out_path,
      {"--sources",
       runfiles->Rlocation("io_bazel/src/tools/singlejar/libtest1.jar").c_str(),
       kPathLibData1, "--include_prefixes", "tools/singlejar/data"});
  std::vector<string> expected_entries(
      {"META-INF/", "META-INF/MANIFEST.MF", "build-data.properties",
       "tools/singlejar/data/", "tools/singlejar/data/extra_file1",
       "tools/singlejar/data/extra_file2"});
  std::vector<string> jar_entries;
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    jar_entries.push_back(cdh->file_name_string());
  }
  input_jar.Close();
  EXPECT_EQ(expected_entries, jar_entries);
}

// --normalize
TEST_F(OutputJarSimpleTest, Normalize) {
  // Creates output jar containing entries from all possible sources:
  //  * archives created by java_library rule, by jar tool, by zip
  //  * resource files
  //  * classpath resource files
  //  *
  string out_path = OutputFilePath("out.jar");
  string testjar_path = OutputFilePath("testinput.jar");
  {
    std::string jar_tool_path = runfiles->Rlocation(JAR_TOOL_PATH);
    string textfile_path = CreateTextFile("jar_testinput.txt", "jar_inputtext");
    string classfile_path = CreateTextFile("JarTestInput.class", "Dummy");
    unlink(testjar_path.c_str());
    ASSERT_EQ(
        0, RunCommand(jar_tool_path.c_str(), "-cf", testjar_path.c_str(),
                      textfile_path.c_str(), classfile_path.c_str(), nullptr));
  }

  string testzip_path = OutputFilePath("testinput.zip");
  {
    string textfile_path = CreateTextFile("zip_testinput.txt", "zip_inputtext");
    string classfile_path = CreateTextFile("ZipTestInput.class", "Dummy");
    unlink(testzip_path.c_str());
    ASSERT_EQ(
        0, RunCommand("zip", "-m", testzip_path.c_str(), textfile_path.c_str(),
                      classfile_path.c_str(), nullptr));
  }

  string resource_path = CreateTextFile("resource", "resource_text");
  string cp_resource_path = CreateTextFile("cp_resource", "cp_resource_text");

  // TODO(asmundak): check the following generated entries, too:
  //  * services
  //  * spring.schemas
  //  * spring.handlers
  //  * protobuf.meta
  //  * extra combiner

  CreateOutput(
      out_path,
      {"--normalize", "--sources",
       runfiles->Rlocation("io_bazel/src/tools/singlejar/libtest1.jar").c_str(),
       testjar_path, testzip_path, "--resources", resource_path,
       "--classpath_resources", cp_resource_path});

  // Scan all entries, verify that *.class entries have timestamp
  // 01/01/2010 00:00:02 and the rest have the timestamp of 01/01/2010 00:00:00.
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    string entry_name = cdh->file_name_string();
    EXPECT_EQ(lh->last_mod_file_date(), cdh->last_mod_file_date())
        << entry_name << " modification date";
    EXPECT_EQ(lh->last_mod_file_time(), cdh->last_mod_file_time())
        << entry_name << " modification time";
    EXPECT_EQ(15393, cdh->last_mod_file_date())
        << entry_name << " modification date should be 01/01/2010";
    auto n = entry_name.size() - strlen(".class");
    if (0 == strcmp(entry_name.c_str() + n, ".class")) {
      EXPECT_EQ(1, cdh->last_mod_file_time())
          << entry_name
          << " modification time for .class entry should be 00:00:02";
    } else {
      EXPECT_EQ(0, cdh->last_mod_file_time())
          << entry_name
          << " modification time for non .class entry should be 00:00:00";
    }
    // Zip creates Unix timestamps, too. Check that normalization removes them.
    ASSERT_EQ(nullptr, cdh->unix_time_extra_field())
        << entry_name << ": CDH should not have Unix Time extra field";
    ASSERT_EQ(nullptr, lh->unix_time_extra_field())
        << entry_name << ": LH should not have Unix Time extra field";
  }
  input_jar.Close();
}

// The files names META-INF/services/<something> are concatenated.
// The files named META-INF/spring.handlers are concatenated.
// The files named META-INF/spring.schemas are concatenated.
TEST_F(OutputJarSimpleTest, Services) {
  CreateTextFile("META-INF/services/spi.DateProvider",
                 "my.DateProviderImpl1\n");
  CreateTextFile("META-INF/services/spi.TimeProvider",
                 "my.TimeProviderImpl1\n");
  CreateTextFile("META-INF/spring.handlers", "handler1\n");
  CreateTextFile("META-INF/spring.schemas", "schema1\n");

  // We have to be in the output directory if we want to have entries in the
  // archive to start with META-INF. The resulting zip will contain 4 entries:
  //   META-INF/services/spi.DateProvider
  //   META-INF/services/spi.TimeProvider
  //   META-INF/spring.handlers
  //   META-INF/spring.schemas
  string out_dir = OutputFilePath("");
  ASSERT_EQ(0,
              RunCommand("cd", out_dir.c_str(), ";",
                         "zip", "-mr", "testinput1.zip", "META-INF", nullptr));
  string zip1_path = OutputFilePath("testinput1.zip");

  // Create the second zip, with 3 files:
  //   META-INF/services/spi.DateProvider.
  //   META-INF/spring.handlers
  //   META-INF/spring.schemas
  CreateTextFile("META-INF/services/spi.DateProvider",
                 "my.DateProviderImpl2\n");
  CreateTextFile("META-INF/spring.handlers", "handler2\n");
  CreateTextFile("META-INF/spring.schemas", "schema2\n");
  ASSERT_EQ(0,
              RunCommand("cd ", out_dir.c_str(), ";",
                         "zip", "-mr", "testinput2.zip", "META-INF", nullptr));
  string zip2_path = OutputFilePath("testinput2.zip");

  // The output jar should contain two service entries. The contents of the
  // META-INF/services/spi.DateProvider should be the concatenation of the
  // contents of this entry from both archives. And it should also contain
  // spring.handlers and spring.schemas entries.
  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--sources", zip1_path, zip2_path});
  EXPECT_EQ("my.DateProviderImpl1\n" "my.DateProviderImpl2\n",
            GetEntryContents(out_path, "META-INF/services/spi.DateProvider"));
  EXPECT_EQ("my.TimeProviderImpl1\n",
            GetEntryContents(out_path, "META-INF/services/spi.TimeProvider"));

  EXPECT_EQ("schema1\n" "schema2\n",
            GetEntryContents(out_path, "META-INF/spring.schemas"));
  EXPECT_EQ("handler1\n" "handler2\n",
            GetEntryContents(out_path, "META-INF/spring.handlers"));
}

// Test that in the absence of the compression option all the plain files in
// the output archive are not compressed but just stored.
TEST_F(OutputJarSimpleTest, NoCompressionOption) {
  string out_path = CompressionOptionsTestingJar("");
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    string entry_name = lh->file_name_string();
    // Each file entry is compressed, each directory entry is uncompressed.
    EXPECT_EQ(lh->compression_method(), cdh->compression_method());
    EXPECT_EQ(Z_NO_COMPRESSION, lh->compression_method())
        << "Entry " << entry_name << " should be stored.";
  }
  input_jar.Close();
}

// Test --compression option. If enabled, all file entries are compressed
// while all directory entries remain uncompressed.
TEST_F(OutputJarSimpleTest, CompressionOption) {
  string out_path = CompressionOptionsTestingJar("--compression");
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    string entry_name = lh->file_name_string();
    // Each file entry is compressed, each directory entry is uncompressed.
    EXPECT_EQ(lh->compression_method(), cdh->compression_method());
    if (lh->file_name()[lh->file_name_length() - 1] != '/') {
      EXPECT_EQ(Z_DEFLATED, lh->compression_method())
          << "File entry " << entry_name << " should be compressed.";
    } else {
      EXPECT_EQ(Z_NO_COMPRESSION, lh->compression_method())
          << "Directory entry " << entry_name << " should be stored.";
    }
  }
  input_jar.Close();
}

// Test --dontchangecompression option. If enabled, existing file entries are
// copied as is, and created entries are compressed.
// Test --compression option. If enabled, all file entries are compressed
// while all directory entries remain uncompressed.
TEST_F(OutputJarSimpleTest, DontChangeCompressionOption) {
  string out_path = CompressionOptionsTestingJar("--dont_change_compression");
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  std::string kStoredEntry =
      runfiles->Rlocation("io_bazel/src/tools/singlejar/output_jar.cc");

  while ((cdh = input_jar.NextEntry(&lh))) {
    string entry_name = lh->file_name_string();
    EXPECT_EQ(lh->compression_method(), cdh->compression_method());
    if (lh->file_name()[lh->file_name_length() - 1] != '/') {
      // All created file entries are compressed, and so are all the file
      // entries from the input jar created by the java_library rule. Only
      // the file entries from the 'stored_jar' should be uncompressed, and
      // it contains a single one:
      if (entry_name == kStoredEntry) {
        EXPECT_EQ(Z_NO_COMPRESSION, lh->compression_method())
            << "File entry " << entry_name << " should be stored.";
      } else {
        EXPECT_EQ(Z_DEFLATED, lh->compression_method())
            << "File entry " << entry_name << " should be compressed.";
      }
    } else {
      EXPECT_EQ(Z_NO_COMPRESSION, lh->compression_method())
          << "Directory entry " << entry_name << " should be stored.";
    }
  }
  input_jar.Close();
}

const char kBuildDataFile[] = "build-data.properties";

// Test --exclude_build_data option when none of the source archives contain
// build-data.properties file: no such file in the output archive.
TEST_F(OutputJarSimpleTest, ExcludeBuildData1) {
  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--exclude_build_data"});
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    string entry_name = lh->file_name_string();
    EXPECT_NE(kBuildDataFile, lh->file_name_string());
  }
  input_jar.Close();
}

// Test --exclude_build_data option when a source archive contains
// build-data.properties file, it should be then copied to the output.
TEST_F(OutputJarSimpleTest, ExcludeBuildData2) {
  string out_dir = OutputFilePath("");
  string testzip_path = OutputFilePath("testinput.zip");
  string buildprop_path = CreateTextFile(kBuildDataFile, "build: foo");
  unlink(testzip_path.c_str());
  ASSERT_EQ(0, RunCommand("cd ", out_dir.c_str(), ";", "zip", "-m",
                          "testinput.zip", kBuildDataFile , nullptr));
  string out_path = OutputFilePath("out.jar");
  CreateOutput(out_path, {"--exclude_build_data", "--sources", testzip_path});
  EXPECT_EQ("build: foo", GetEntryContents(out_path, kBuildDataFile));
}

// Test that the entries with suffixes in --nocompressed_suffixes are
// not compressed. This applies both to the source archives' entries and
// standalone files.
TEST_F(OutputJarSimpleTest, Nocompress) {
  string res1_path =
      CreateTextFile("resource.foo", "line1\nline2\nline3\nline4\n");
  string res2_path =
      CreateTextFile("resource.bar", "line1\nline2\nline3\nline4\n");
  string out_path = OutputFilePath("out.jar");
  CreateOutput(
      out_path,
      {"--compression", "--sources",
       runfiles->Rlocation("io_bazel/src/tools/singlejar/libtest1.jar").c_str(),
       "--resources", res1_path, res2_path, "--nocompress_suffixes", ".foo",
       ".h"});
  InputJar input_jar;
  ASSERT_TRUE(input_jar.Open(out_path));
  const LH *lh;
  const CDH *cdh;
  while ((cdh = input_jar.NextEntry(&lh))) {
    const char *entry_name_end = lh->file_name() + lh->file_name_length();
    if (!strncmp(entry_name_end - 4, ".foo", 4) ||
        !strncmp(entry_name_end - 2, ".h", 2)) {
      EXPECT_EQ(Z_NO_COMPRESSION, lh->compression_method())
          << "Expected " << lh->file_name_string() << " uncompressed";
    } else if (!strncmp(entry_name_end - 3, ".cc", 3) ||
               !strncmp(entry_name_end - 4, ".bar", 4)) {
      EXPECT_EQ(Z_DEFLATED, lh->compression_method())
          << "Expected " << lh->file_name_string() << " compressed";
    }
  }
  input_jar.Close();
}

}  // namespace
