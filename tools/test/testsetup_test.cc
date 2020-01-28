#include <stdio.h>
#include <sys/stat.h>
#include "googletest/include/gtest/gtest.h"

// Test helper functions

// Two constants used for path-type checks
#define OPTIONAL (true)
#define MUST_EXIST (false)

// Return true if the path is a directory; if optional,
// then returns true if the path is NULL or doesn't exist
static bool IsDirectory(const char *path, bool optional) {
  if(path == NULL) {
    return optional;
  }

  struct stat st;
  if (stat(path, &st) < 0) {
    return optional;
  }
  return (st.st_mode & S_IFDIR) == S_IFDIR;
} 


// Return true if the path is a regular file; if optional,
// then returns true if the path is NULL or doesn't exist
static bool IsFile(const char *path, bool optional) {
  if(path == NULL) {
    return optional;
  }

  struct stat st;
  if (stat(path, &st) < 0) {
    return optional;
  }
  return (st.st_mode & S_IFREG) == S_IFREG;
}

// Verify the test environment as described by the Test Encyclopedia
// <https://docs.bazel.build/versions/master/test-encyclopedia.html>
TEST(TestSetup, Env) {
  // Test values in same order as in the Test Encyclopedia.

  // TEST_SRCDIR and TEST_TMPDIR are tested out-of-order as several
  // other values are based on their values.
  char *testSrcDir = getenv("TEST_SRCDIR");
  EXPECT_NE(testSrcDir, nullptr);
  EXPECT_TRUE(IsDirectory(testSrcDir, MUST_EXIST));
  char *testTmpDir = getenv("TEST_TMPDIR");
  EXPECT_NE(testTmpDir, nullptr);
  EXPECT_TRUE(IsDirectory(testTmpDir, MUST_EXIST));

  // HOME is recommended, and should be TEST_TMPDIR
  EXPECT_NE(getenv("HOME"), nullptr);
  EXPECT_STREQ(getenv("HOME"), testTmpDir);

  EXPECT_EQ(getenv("LANG"), nullptr);
  EXPECT_EQ(getenv("LANGUAGE"), nullptr);
  EXPECT_EQ(getenv("LC_ALL"), nullptr);
  EXPECT_EQ(getenv("LC_COLLATE"), nullptr);
  EXPECT_EQ(getenv("LC_CTYPE"), nullptr);
  EXPECT_EQ(getenv("LC_MESSAGES"), nullptr);
  EXPECT_EQ(getenv("LC_MONETARY"), nullptr);
  EXPECT_EQ(getenv("LC_NUMERIC"), nullptr);
  EXPECT_EQ(getenv("LC_TIME"), nullptr);

  // LD_LIBRARY_PATH is optional

  // JAVA_RUNFILES is marked as deprecated
  EXPECT_NE(getenv("JAVA_RUNFILES"), nullptr);
  EXPECT_TRUE(IsDirectory(getenv("JAVA_RUNFILES"), MUST_EXIST));
  EXPECT_STREQ(getenv("JAVA_RUNFILES"), testSrcDir);

  // LOGNAME should be set to USER
  EXPECT_NE(getenv("USER"), nullptr);
  EXPECT_NE(getenv("LOGNAME"), nullptr);
  EXPECT_STREQ(getenv("LOGNAME"), getenv("USER"));

  EXPECT_NE(getenv("PATH"), nullptr);

  // PWD is recommended, but is only set on *nix systems, and should
  // be $TEST_SRCDIR/workspace-name
  char *pwd = getenv("PWD");
  EXPECT_TRUE(IsDirectory(pwd, OPTIONAL));

  EXPECT_STREQ(getenv("SHLVL"), "2");
  EXPECT_TRUE(IsFile(getenv("TEST_PREMATURE_EXIT_FILE"), OPTIONAL));
  // TEST_RANDOM_SEED is optional
  // TEST_TARGET is optional
  // TEST_SIZE is optional
  // TEST_TIMEOUT is optional
  // TEST_SHARD_INDEX is optional
  EXPECT_TRUE(IsFile(getenv("TEST_SHARD_STATUS_FILE"), OPTIONAL));
  // TEST_SRCDIR already tested
  // TEST_TOTAL_SHARDS is optional
  // TEST_TMPDIR already tested
  EXPECT_TRUE(IsFile(getenv("TEST_WARNINGS_OUTPUT_FILE"), OPTIONAL));
  // TESTBRIDGE_TEST_ONLY: is optional
  EXPECT_STREQ(getenv("TZ"), "UTC");
  // USER already tested
  EXPECT_TRUE(IsFile(getenv("XML_OUTPUT_FILE"), OPTIONAL));
  EXPECT_NE(getenv("TEST_WORKSPACE"), nullptr);
}
