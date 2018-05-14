// Copyright 2018 The Bazel Authors. All rights reserved.
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

#include "tools/cpp/runfiles/runfiles.h"

#ifdef COMPILER_MSVC
#include <windows.h>
#endif  // COMPILER_MSVC

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/main/cpp/util/file.h"

#define _T(x) #x
#define T(x) _T(x)
#define LINE() T(__LINE__)

namespace bazel {
namespace tools {
namespace cpp {
namespace runfiles {
namespace {

using bazel::tools::cpp::runfiles::testing::TestOnly_CreateRunfiles;
using bazel::tools::cpp::runfiles::testing::TestOnly_IsAbsolute;
using std::cerr;
using std::endl;
using std::function;
using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

class RunfilesTest : public ::testing::Test {
 protected:
  // Create a temporary file that is deleted with the destructor.
  class MockFile {
   public:
    // Create an empty file with the given name under $TEST_TMPDIR.
    static MockFile* Create(const string& name);

    // Create a file with the given name and contents under $TEST_TMPDIR.
    // The method ensures to create all parent directories, so `name` is allowed
    // to contain directory components.
    static MockFile* Create(const string& name, const vector<string>& lines);

    ~MockFile();
    const string& Path() const { return path_; }

   private:
    MockFile(const string& path) : path_(path) {}
    MockFile(const MockFile&) = delete;
    MockFile(MockFile&&) = delete;
    MockFile& operator=(const MockFile&) = delete;
    MockFile& operator=(MockFile&&) = delete;

    const string path_;
  };

  static string GetTemp();

  static function<string(const string&)> kEnvWithTestSrcdir;
};

function<string(const string&)> RunfilesTest::kEnvWithTestSrcdir =
    [](const string& key) {
      if (key == "TEST_SRCDIR") {
        return string("always ignored");
      } else {
        return string();
      }
    };

string RunfilesTest::GetTemp() {
#ifdef COMPILER_MSVC
  DWORD size = ::GetEnvironmentVariableA("TEST_TMPDIR", NULL, 0);
  if (size == 0) {
    return std::move(string());  // unset or empty envvar
  }
  unique_ptr<char[]> value(new char[size]);
  ::GetEnvironmentVariableA("TEST_TMPDIR", value.get(), size);
  return std::move(string(value.get()));
#else
  char* result = getenv("TEST_TMPDIR");
  return result != NULL ? std::move(string(result)) : std::move(string());
#endif
}

RunfilesTest::MockFile* RunfilesTest::MockFile::Create(const string& name) {
  return Create(name, vector<string>());
}

RunfilesTest::MockFile* RunfilesTest::MockFile::Create(
    const string& name, const vector<string>& lines) {
  if (name.find("..") != string::npos || TestOnly_IsAbsolute(name)) {
    cerr << "WARNING: " << __FILE__ << "(" << __LINE__ << "): bad name: \""
         << name << "\"" << endl;
    return nullptr;
  }

  string tmp(std::move(RunfilesTest::GetTemp()));
  if (tmp.empty()) {
    cerr << "WARNING: " << __FILE__ << "(" << __LINE__
         << "): $TEST_TMPDIR is empty" << endl;
    return nullptr;
  }
  string path(tmp + "/" + name);
  string dirname = blaze_util::Dirname(path);
  if (!blaze_util::MakeDirectories(dirname, 0777)) {
    cerr << "WARNING: " << __FILE__ << "(" << __LINE__ << "): MakeDirectories("
         << dirname << ") failed" << endl;
    return nullptr;
  }

  auto stm = std::ofstream(path);
  for (auto i : lines) {
    stm << i << std::endl;
  }
  return new MockFile(path);
}

RunfilesTest::MockFile::~MockFile() { std::remove(path_.c_str()); }

TEST_F(RunfilesTest, CreatesManifestBasedRunfilesFromManifestNextToBinary) {
  unique_ptr<MockFile> mf(
      MockFile::Create("foo" LINE() ".runfiles_manifest", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);
  string argv0(mf->Path().substr(
      0, mf->Path().size() - string(".runfiles_manifest").size()));

  string error;
  unique_ptr<Runfiles> r(
      TestOnly_CreateRunfiles(argv0, kEnvWithTestSrcdir, &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  // We know it's manifest-based because it returns empty string for unknown
  // paths.
  EXPECT_EQ(r->Rlocation("unknown"), "");
}

TEST_F(RunfilesTest,
       CreatesManifestBasedRunfilesFromManifestInRunfilesDirectory) {
  unique_ptr<MockFile> mf(
      MockFile::Create("foo" LINE() ".runfiles/MANIFEST", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);
  string argv0(mf->Path().substr(
      0, mf->Path().size() - string(".runfiles/MANIFEST").size()));

  string error;
  unique_ptr<Runfiles> r(
      TestOnly_CreateRunfiles(argv0, kEnvWithTestSrcdir, &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  // We know it's manifest-based because it returns empty string for unknown
  // paths.
  EXPECT_EQ(r->Rlocation("unknown"), "");
}

TEST_F(RunfilesTest, CreatesManifestBasedRunfilesFromEnvvar) {
  unique_ptr<MockFile> mf(
      MockFile::Create("foo" LINE() ".runfiles_manifest", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(TestOnly_CreateRunfiles(
      "ignore-argv0",
      [&mf](const string& key) {
        if (key == "RUNFILES_MANIFEST_FILE") {
          return mf->Path();
        } else if (key == "RUNFILES_DIR") {
          return string("ignored when RUNFILES_MANIFEST_FILE has a value");
        } else if (key == "TEST_SRCDIR") {
          return string("always ignored");
        } else {
          return string();
        }
      },
      &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  // We know it's manifest-based because it returns empty string for unknown
  // paths.
  EXPECT_EQ(r->Rlocation("unknown"), "");
}

TEST_F(RunfilesTest, CannotCreateManifestBasedRunfilesDueToBadManifest) {
  unique_ptr<MockFile> mf(
      MockFile::Create("foo" LINE() ".runfiles_manifest", {"a b", "nospace"}));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(Runfiles::CreateManifestBased(mf->Path(), &error));
  ASSERT_EQ(r, nullptr);
  EXPECT_NE(error.find("bad runfiles manifest entry"), string::npos);
  EXPECT_NE(error.find("line #2: \"nospace\""), string::npos);
}

TEST_F(RunfilesTest, ManifestBasedRunfilesRlocation) {
  unique_ptr<MockFile> mf(
      MockFile::Create("foo" LINE() ".runfiles_manifest", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(Runfiles::CreateManifestBased(mf->Path(), &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  EXPECT_EQ(r->Rlocation("c/d"), "");
  EXPECT_EQ(r->Rlocation(""), "");
  EXPECT_EQ(r->Rlocation("foo"), "");
  EXPECT_EQ(r->Rlocation("foo/"), "");
  EXPECT_EQ(r->Rlocation("foo/bar"), "");
  EXPECT_EQ(r->Rlocation("../foo"), "");
  EXPECT_EQ(r->Rlocation("foo/.."), "");
  EXPECT_EQ(r->Rlocation("foo/../bar"), "");
  EXPECT_EQ(r->Rlocation("./foo"), "");
  EXPECT_EQ(r->Rlocation("foo/."), "");
  EXPECT_EQ(r->Rlocation("foo/./bar"), "");
  EXPECT_EQ(r->Rlocation("//foo"), "");
  EXPECT_EQ(r->Rlocation("foo//"), "");
  EXPECT_EQ(r->Rlocation("foo//bar"), "");
  EXPECT_EQ(r->Rlocation("/Foo"), "/Foo");
  EXPECT_EQ(r->Rlocation("c:/Foo"), "c:/Foo");
  EXPECT_EQ(r->Rlocation("c:\\Foo"), "c:\\Foo");
}

TEST_F(RunfilesTest, DirectoryBasedRunfilesRlocation) {
  string error;
  unique_ptr<Runfiles> r(Runfiles::CreateDirectoryBased("whatever", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(r->Rlocation("a/b"), "whatever/a/b");
  EXPECT_EQ(r->Rlocation("c/d"), "whatever/c/d");
  EXPECT_EQ(r->Rlocation(""), "");
  EXPECT_EQ(r->Rlocation("foo"), "whatever/foo");
  EXPECT_EQ(r->Rlocation("foo/"), "whatever/foo/");
  EXPECT_EQ(r->Rlocation("foo/bar"), "whatever/foo/bar");
  EXPECT_EQ(r->Rlocation("../foo"), "");
  EXPECT_EQ(r->Rlocation("foo/.."), "");
  EXPECT_EQ(r->Rlocation("foo/../bar"), "");
  EXPECT_EQ(r->Rlocation("./foo"), "");
  EXPECT_EQ(r->Rlocation("foo/."), "");
  EXPECT_EQ(r->Rlocation("foo/./bar"), "");
  EXPECT_EQ(r->Rlocation("//foo"), "");
  EXPECT_EQ(r->Rlocation("foo//"), "");
  EXPECT_EQ(r->Rlocation("foo//bar"), "");
  EXPECT_EQ(r->Rlocation("/Foo"), "/Foo");
  EXPECT_EQ(r->Rlocation("c:/Foo"), "c:/Foo");
  EXPECT_EQ(r->Rlocation("c:\\Foo"), "c:\\Foo");
}

TEST_F(RunfilesTest, ManifestBasedRunfilesEnvVars) {
  const vector<string> suffixes({"/MANIFEST", ".runfiles_manifest",
                                 "runfiles_manifest", ".runfiles", ".manifest",
                                 ".txt"});
  for (vector<string>::size_type i = 0; i < suffixes.size(); ++i) {
    unique_ptr<MockFile> mf(
        MockFile::Create(string("foo" LINE()) + suffixes[i]));
    EXPECT_TRUE(mf != nullptr) << " (suffix=\"" << suffixes[i] << "\")";

    string error;
    unique_ptr<Runfiles> r(Runfiles::CreateManifestBased(mf->Path(), &error));
    ASSERT_NE(r, nullptr) << " (suffix=\"" << suffixes[i] << "\")";
    EXPECT_TRUE(error.empty());

    // The object can compute the runfiles directory when i=0 and i=1, but not
    // when i>1 because the manifest file's name doesn't end in a well-known
    // way.
    const string expected_runfiles_dir(
        i < 2 ? mf->Path().substr(0, mf->Path().size() - 9 /* "_manifest" */)
              : "");
    vector<pair<string, string> > expected(
        {{"RUNFILES_MANIFEST_FILE", mf->Path()},
         {"JAVA_RUNFILES", expected_runfiles_dir}});
    EXPECT_EQ(r->EnvVars(), expected) << " (suffix=\"" << suffixes[i] << "\")";
  }
}

TEST_F(RunfilesTest, CreatesDirectoryBasedRunfilesFromDirectoryNextToBinary) {
  // We create a directory as a side-effect of creating a mock file.
  unique_ptr<MockFile> mf(
      MockFile::Create(string("foo" LINE() ".runfiles/dummy")));
  string argv0(mf->Path().substr(
      0, mf->Path().size() - string(".runfiles/dummy").size()));

  string error;
  unique_ptr<Runfiles> r(
      TestOnly_CreateRunfiles(argv0, kEnvWithTestSrcdir, &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(r->Rlocation("a/b"), argv0 + ".runfiles/a/b");
  // We know it's directory-based because it returns some result for unknown
  // paths.
  EXPECT_EQ(r->Rlocation("unknown"), argv0 + ".runfiles/unknown");
}

TEST_F(RunfilesTest, CreatesDirectoryBasedRunfilesFromEnvvar) {
  string error;
  unique_ptr<Runfiles> r(
      TestOnly_CreateRunfiles("ignore-argv0",
                              [](const string& key) {
                                if (key == "RUNFILES_DIR") {
                                  return string("runfiles/dir");
                                } else if (key == "TEST_SRCDIR") {
                                  return string("always ignored");
                                } else {
                                  return string();
                                }
                              },
                              &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(r->Rlocation("a/b"), "runfiles/dir/a/b");
  EXPECT_EQ(r->Rlocation("foo"), "runfiles/dir/foo");
  EXPECT_EQ(r->Rlocation("/Foo"), "/Foo");
  EXPECT_EQ(r->Rlocation("c:/Foo"), "c:/Foo");
  EXPECT_EQ(r->Rlocation("c:\\Foo"), "c:\\Foo");
}

TEST_F(RunfilesTest, DirectoryBasedRunfilesEnvVars) {
  string error;
  unique_ptr<Runfiles> r(
      Runfiles::CreateDirectoryBased("runfiles/dir", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  vector<pair<string, string> > expected(
      {{"RUNFILES_DIR", "runfiles/dir"}, {"JAVA_RUNFILES", "runfiles/dir"}});
  EXPECT_EQ(r->EnvVars(), expected);
}

TEST_F(RunfilesTest, FailsToCreateManifestBasedBecauseManifestDoesNotExist) {
  string error;
  unique_ptr<Runfiles> r(
      Runfiles::CreateManifestBased("non-existent-file", &error));
  ASSERT_EQ(r, nullptr);
  EXPECT_NE(error.find("cannot open runfiles manifest"), string::npos);
}

TEST_F(RunfilesTest, FailsToCreateAnyRunfilesBecauseEnvvarsAreNotDefined) {
  unique_ptr<MockFile> mf(MockFile::Create(string("foo" LINE())));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(
      TestOnly_CreateRunfiles("ignore-argv0",
                              [&mf](const string& key) {
                                if (key == "RUNFILES_MANIFEST_FILE") {
                                  return mf->Path();
                                } else if (key == "RUNFILES_DIR") {
                                  return string("whatever");
                                } else if (key == "TEST_SRCDIR") {
                                  return string("always ignored");
                                } else {
                                  return string();
                                }
                              },
                              &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  r.reset(TestOnly_CreateRunfiles("ignore-argv0",
                                  [](const string& key) {
                                    if (key == "RUNFILES_DIR") {
                                      return string("whatever");
                                    } else if (key == "TEST_SRCDIR") {
                                      return string("always ignored");
                                    } else {
                                      return string();
                                    }
                                  },
                                  &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  r.reset(TestOnly_CreateRunfiles("ignore-argv0", kEnvWithTestSrcdir, &error));
  ASSERT_EQ(r, nullptr);
  EXPECT_NE(error.find("cannot find runfiles"), string::npos);
}

TEST_F(RunfilesTest, MockFileTest) {
  {
    unique_ptr<MockFile> mf(MockFile::Create(string("foo" LINE() "/..")));
    EXPECT_TRUE(mf == nullptr);
  }

  {
    unique_ptr<MockFile> mf(MockFile::Create(string("/Foo" LINE())));
    EXPECT_TRUE(mf == nullptr);
  }

  {
    unique_ptr<MockFile> mf(MockFile::Create(string("C:/Foo" LINE())));
    EXPECT_TRUE(mf == nullptr);
  }

  string path;
  {
    unique_ptr<MockFile> mf(MockFile::Create(string("foo" LINE() "/bar1/qux")));
    EXPECT_TRUE(mf != nullptr);
    path = mf->Path();

    std::ifstream stm(path);
    EXPECT_TRUE(stm.good());
    string actual;
    stm >> actual;
    EXPECT_TRUE(actual.empty());
  }
  {
    std::ifstream stm(path);
    EXPECT_FALSE(stm.good());
  }

  {
    unique_ptr<MockFile> mf(
        MockFile::Create(string("foo" LINE() "/bar2/qux"), vector<string>()));
    EXPECT_TRUE(mf != nullptr);
    path = mf->Path();

    std::ifstream stm(path);
    EXPECT_TRUE(stm.good());
    string actual;
    stm >> actual;
    EXPECT_TRUE(actual.empty());
  }
  {
    std::ifstream stm(path);
    EXPECT_FALSE(stm.good());
  }

  {
    unique_ptr<MockFile> mf(
        MockFile::Create(string("foo" LINE() "/bar3/qux"),
                         {"hello world", "you are beautiful"}));
    EXPECT_TRUE(mf != nullptr);
    path = mf->Path();

    std::ifstream stm(path);
    EXPECT_TRUE(stm.good());
    string actual;
    std::getline(stm, actual);
    EXPECT_EQ("hello world", actual);
    std::getline(stm, actual);
    EXPECT_EQ("you are beautiful", actual);
    std::getline(stm, actual);
    EXPECT_EQ("", actual);
  }
  {
    std::ifstream stm(path);
    EXPECT_FALSE(stm.good());
  }
}

TEST_F(RunfilesTest, IsAbsolute) {
  EXPECT_FALSE(TestOnly_IsAbsolute("foo"));
  EXPECT_FALSE(TestOnly_IsAbsolute("foo/bar"));
  EXPECT_FALSE(TestOnly_IsAbsolute("\\foo"));
  EXPECT_TRUE(TestOnly_IsAbsolute("c:\\foo"));
  EXPECT_TRUE(TestOnly_IsAbsolute("c:/foo"));
  EXPECT_TRUE(TestOnly_IsAbsolute("/foo"));
  EXPECT_TRUE(TestOnly_IsAbsolute("x:\\foo"));
  EXPECT_FALSE(TestOnly_IsAbsolute("::\\foo"));
  EXPECT_FALSE(TestOnly_IsAbsolute("x\\foo"));
  EXPECT_FALSE(TestOnly_IsAbsolute("x:"));
  EXPECT_TRUE(TestOnly_IsAbsolute("x:\\"));
}

TEST_F(RunfilesTest, PathsFromEnvVars) {
  string mf, dir;

  static const function<string(string)> kEnvVars = [](string key) {
    if (key == "TEST_SRCDIR") {
      return "always ignored";
    } else if (key == "RUNFILES_MANIFEST_FILE") {
      return "mock1/MANIFEST";
    } else if (key == "RUNFILES_DIR") {
      return "mock2";
    } else {
      return "";
    }
  };

  // Both envvars have a valid value.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars,
      [](const string& path) { return path == "mock1/MANIFEST"; },
      [](const string& path) { return path == "mock2"; }, &mf, &dir));
  EXPECT_EQ(mf, "mock1/MANIFEST");
  EXPECT_EQ(dir, "mock2");

  // RUNFILES_MANIFEST_FILE is invalid but RUNFILES_DIR is good and there's a
  // runfiles manifest in the runfiles directory.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars,
      [](const string& path) { return path == "mock2/MANIFEST"; },
      [](const string& path) { return path == "mock2"; }, &mf, &dir));
  EXPECT_EQ(mf, "mock2/MANIFEST");
  EXPECT_EQ(dir, "mock2");

  // RUNFILES_MANIFEST_FILE is invalid but RUNFILES_DIR is good, but there's no
  // runfiles manifest in the runfiles directory.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars, [](const string& path) { return false; },
      [](const string& path) { return path == "mock2"; }, &mf, &dir));
  EXPECT_EQ(mf, "");
  EXPECT_EQ(dir, "mock2");

  // RUNFILES_DIR is invalid but RUNFILES_MANIFEST_FILE is good, and it is in
  // a valid-looking runfiles directory.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars,
      [](const string& path) { return path == "mock1/MANIFEST"; },
      [](const string& path) { return path == "mock1"; }, &mf, &dir));
  EXPECT_EQ(mf, "mock1/MANIFEST");
  EXPECT_EQ(dir, "mock1");

  // RUNFILES_DIR is invalid but RUNFILES_MANIFEST_FILE is good, but it is not
  // in any valid-looking runfiles directory.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars,
      [](const string& path) { return path == "mock1/MANIFEST"; },
      [](const string& path) { return false; }, &mf, &dir));
  EXPECT_EQ(mf, "mock1/MANIFEST");
  EXPECT_EQ(dir, "");

  // Both envvars are invalid, but there's a manifest in a runfiles directory
  // next to argv0, however there's no other content in the runfiles directory.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars,
      [](const string& path) { return path == "argv0.runfiles/MANIFEST"; },
      [](const string& path) { return false; }, &mf, &dir));
  EXPECT_EQ(mf, "argv0.runfiles/MANIFEST");
  EXPECT_EQ(dir, "");

  // Both envvars are invalid, but there's a manifest next to argv0. There's
  // no runfiles tree anywhere.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars,
      [](const string& path) { return path == "argv0.runfiles_manifest"; },
      [](const string& path) { return false; }, &mf, &dir));
  EXPECT_EQ(mf, "argv0.runfiles_manifest");
  EXPECT_EQ(dir, "");

  // Both envvars are invalid, but there's a valid manifest next to argv0, and a
  // valid runfiles directory (without a manifest in it).
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars,
      [](const string& path) { return path == "argv0.runfiles_manifest"; },
      [](const string& path) { return path == "argv0.runfiles"; }, &mf, &dir));
  EXPECT_EQ(mf, "argv0.runfiles_manifest");
  EXPECT_EQ(dir, "argv0.runfiles");

  // Both envvars are invalid, but there's a valid runfiles directory next to
  // argv0, though no manifest in it.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars, [](const string& path) { return false; },
      [](const string& path) { return path == "argv0.runfiles"; }, &mf, &dir));
  EXPECT_EQ(mf, "");
  EXPECT_EQ(dir, "argv0.runfiles");

  // Both envvars are invalid, but there's a valid runfiles directory next to
  // argv0 with a valid manifest in it.
  EXPECT_TRUE(Runfiles::PathsFrom(
      "argv0", kEnvVars,
      [](const string& path) { return path == "argv0.runfiles/MANIFEST"; },
      [](const string& path) { return path == "argv0.runfiles"; }, &mf, &dir));
  EXPECT_EQ(mf, "argv0.runfiles/MANIFEST");
  EXPECT_EQ(dir, "argv0.runfiles");
}

}  // namespace
}  // namespace runfiles
}  // namespace cpp
}  // namespace tools
}  // namespace bazel
