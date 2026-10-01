// Copyright 2026 The Bazel Authors. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <string_view>
#include <vector>

#include "third_party/ijar/common.h"
#include "third_party/ijar/zip.h"

#define IJAR_TEST_CHECK(cond)                                            \
  do {                                                                   \
    if (!(cond)) {                                                       \
      fprintf(stderr, "IJAR_TEST_CHECK failed at %s:%d: %s\n", __FILE__, \
              __LINE__, #cond);                                          \
      abort();                                                           \
    }                                                                    \
  } while (0)

namespace devtools_ijar {
namespace {

// Helper to build big-endian class file byte vectors.
class ByteBuilder {
 public:
  void u1_val(u1 v) { buf_.push_back(v); }
  void u2be(u2 v) {
    buf_.push_back(static_cast<u1>((v >> 8) & 0xff));
    buf_.push_back(static_cast<u1>(v & 0xff));
  }
  void u4be(u4 v) {
    buf_.push_back(static_cast<u1>((v >> 24) & 0xff));
    buf_.push_back(static_cast<u1>((v >> 16) & 0xff));
    buf_.push_back(static_cast<u1>((v >> 8) & 0xff));
    buf_.push_back(static_cast<u1>(v & 0xff));
  }
  void u2le(u2 v) {
    buf_.push_back(static_cast<u1>(v & 0xff));
    buf_.push_back(static_cast<u1>((v >> 8) & 0xff));
  }
  void u4le(u4 v) {
    buf_.push_back(static_cast<u1>(v & 0xff));
    buf_.push_back(static_cast<u1>((v >> 8) & 0xff));
    buf_.push_back(static_cast<u1>((v >> 16) & 0xff));
    buf_.push_back(static_cast<u1>((v >> 24) & 0xff));
  }
  void u8le(u8 v) {
    u4le(static_cast<u4>(v & 0xffffffffULL));
    u4le(static_cast<u4>((v >> 32) & 0xffffffffULL));
  }
  void utf8_const(std::string_view s) {
    u1_val(1);  // CONSTANT_Utf8
    u2be(static_cast<u2>(s.size()));
    for (char c : s) {
      u1_val(static_cast<u1>(c));
    }
  }
  void class_const(u2 name_idx) {
    u1_val(7);  // CONSTANT_Class
    u2be(name_idx);
  }
  const std::vector<u1>& bytes() const { return buf_; }

 private:
  std::vector<u1> buf_;
};

// Runs StripClass on input bytes and verifies it does not crash or overflow.
// For malformed classfiles, verifies that StripClass fell back to copying
// verbatim.
void RunStripClass(const std::vector<u1>& in) {
  std::vector<u1> out(in.size() + 4096, 0);
  u1* out_p = out.data();
  bool keep = StripClass(out_p, in.data(), in.size());
  IJAR_TEST_CHECK(keep);
  size_t written = static_cast<size_t>(out_p - out.data());
  IJAR_TEST_CHECK(written == in.size());
  if (!in.empty()) {
    IJAR_TEST_CHECK(memcmp(out.data(), in.data(), in.size()) == 0);
  }
}

void TestTruncatedMagicAndHeader() {
  // Empty and tiny inputs
  RunStripClass({});
  RunStripClass({0xCA, 0xFE});
  RunStripClass({0xCA, 0xFE, 0xBA, 0xBE});  // magic only, no version/cp count
  RunStripClass({0xCA, 0xFE, 0xBA, 0xBE, 0x00, 0x00, 0x00, 0x3D});
  printf("PASS: TestTruncatedMagicAndHeader\n");
}

// Regression test for PR #29739: truncated constant pool & Utf8 length past
// EOF.
void TestTruncatedConstantPoolAndUtf8Overflow() {
  // Case 1: constant_pool_count = 20, but EOF after 1 entry
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(20);  // claims 19 entries
    b.utf8_const("Foo");
    RunStripClass(b.bytes());
  }
  // Case 2: CONSTANT_Utf8 claiming 65535 bytes with only 3 bytes remaining
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(2);
    b.u1_val(1);     // CONSTANT_Utf8
    b.u2be(0xFFFF);  // length = 65535
    b.u1_val('a');
    b.u1_val('b');
    b.u1_val('c');
    RunStripClass(b.bytes());
  }
  // Case 3: Unknown constant pool tag
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(2);
    b.u1_val(99);  // invalid tag
    RunStripClass(b.bytes());
  }
  printf("PASS: TestTruncatedConstantPoolAndUtf8Overflow\n");
}

void TestConstantPoolDeadSlotAndWrongTag() {
  // Slot #1 is CONSTANT_Long (takes slots #1 and #2). Slot #2 is dead NULL
  // slot. this_class references dead slot #2.
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(4);
    // #1: Long (tag 5)
    b.u1_val(5);
    b.u4be(0);
    b.u4be(1);
    // #3: Utf8 "java/lang/Object"
    b.utf8_const("java/lang/Object");
    b.u2be(0x0021);  // access_flags
    b.u2be(2);       // this_class -> dead slot #2!
    b.u2be(0);       // super_class
    b.u2be(0);       // interfaces
    b.u2be(0);       // fields
    b.u2be(0);       // methods
    b.u2be(0);       // attrs
    RunStripClass(b.bytes());
  }
  // this_class points to a CONSTANT_Utf8 instead of CONSTANT_Class
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(2);
    b.utf8_const("Foo");  // #1 is Utf8, not Class
    b.u2be(0x0021);
    b.u2be(1);  // this_class -> #1 (wrong tag)
    b.u2be(0);
    b.u2be(0);
    b.u2be(0);
    b.u2be(0);
    b.u2be(0);
    RunStripClass(b.bytes());
  }
  printf("PASS: TestConstantPoolDeadSlotAndWrongTag\n");
}

void TestConstantPoolCycleNoStackOverflow() {
  // #1: Class(name_index = 2)
  // #2: Class(name_index = 1)
  // Previously caused infinite recursion in Constant::slot() / Display().
  ByteBuilder b;
  b.u4be(0xCAFEBABE);
  b.u2be(0);
  b.u2be(61);
  b.u2be(3);
  b.class_const(2);  // #1 -> #2
  b.class_const(1);  // #2 -> #1
  b.u2be(0x0021);
  b.u2be(1);
  b.u2be(2);
  b.u2be(0);
  b.u2be(0);
  b.u2be(0);
  b.u2be(0);
  RunStripClass(b.bytes());
  printf("PASS: TestConstantPoolCycleNoStackOverflow\n");
}

// Regression test for PR #29840: header & member count overflows.
void TestHeaderAndMemberCountOverflows() {
  // interfaces_count = 0xFFFF with 0 interface bytes
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(3);
    b.utf8_const("Foo");
    b.class_const(1);
    b.u2be(0x0021);
    b.u2be(2);
    b.u2be(0);
    b.u2be(0xFFFF);  // interfaces_count = 65535, EOF immediately
    RunStripClass(b.bytes());
  }
  // methods_count = 0xFFFF with 0 method bytes
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(3);
    b.utf8_const("Foo");
    b.class_const(1);
    b.u2be(0x0021);
    b.u2be(2);
    b.u2be(0);
    b.u2be(0);       // interfaces
    b.u2be(0);       // fields
    b.u2be(0xFFFF);  // methods_count = 65535, EOF immediately
    RunStripClass(b.bytes());
  }
  printf("PASS: TestHeaderAndMemberCountOverflows\n");
}

// Regression test for PR #29804: attribute_length bounds and slice enforcement.
void TestAttributeLengthBoundsAndSlice() {
  // Exceptions attribute with declared length = 2, claiming 500 exceptions
  ByteBuilder b;
  b.u4be(0xCAFEBABE);
  b.u2be(0);
  b.u2be(61);
  b.u2be(6);
  b.utf8_const("Foo");         // #1
  b.class_const(1);            // #2
  b.utf8_const("m");           // #3
  b.utf8_const("()V");         // #4
  b.utf8_const("Exceptions");  // #5
  b.u2be(0x0021);
  b.u2be(2);
  b.u2be(0);
  b.u2be(0);  // interfaces
  b.u2be(0);  // fields
  b.u2be(1);  // 1 method
  b.u2be(0x0001);
  b.u2be(3);
  b.u2be(4);
  b.u2be(1);    // 1 method attribute
  b.u2be(5);    // "Exceptions"
  b.u4be(2);    // declared attribute_length = 2 bytes
  b.u2be(500);  // number_of_exceptions = 500 (would read 1000 bytes without
                // slice!)
  b.u2be(0);    // class attributes_count = 0
  RunStripClass(b.bytes());
  printf("PASS: TestAttributeLengthBoundsAndSlice\n");
}

void TestAnnotationElementValueInvalidTagAndDeepRecursion() {
  // Case 1: invalid element_value tag 'X' (previously triggered abort())
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(5);
    b.utf8_const("Foo");                        // #1
    b.class_const(1);                           // #2
    b.utf8_const("RuntimeVisibleAnnotations");  // #3
    b.utf8_const("LBar;");                      // #4
    b.u2be(0x0021);
    b.u2be(2);
    b.u2be(0);
    b.u2be(0);
    b.u2be(0);
    b.u2be(0);
    b.u2be(1);      // 1 class attr
    b.u2be(3);      // RuntimeVisibleAnnotations
    b.u4be(9);      // attr length
    b.u2be(1);      // 1 annotation
    b.u2be(4);      // type_index "LBar;"
    b.u2be(1);      // 1 element_value pair
    b.u2be(1);      // element_name_index
    b.u1_val('X');  // invalid tag 'X'
    RunStripClass(b.bytes());
  }
  // Case 2: deeply nested array element_value (100 levels deep > 64 limit)
  {
    ByteBuilder b;
    b.u4be(0xCAFEBABE);
    b.u2be(0);
    b.u2be(61);
    b.u2be(5);
    b.utf8_const("Foo");                        // #1
    b.class_const(1);                           // #2
    b.utf8_const("RuntimeVisibleAnnotations");  // #3
    b.utf8_const("LBar;");                      // #4
    b.u2be(0x0021);
    b.u2be(2);
    b.u2be(0);
    b.u2be(0);
    b.u2be(0);
    b.u2be(0);
    b.u2be(1);
    b.u2be(3);
    int depth = 100;
    b.u4be(static_cast<u4>(8 + depth * 3 + 3));
    b.u2be(1);
    b.u2be(4);
    b.u2be(1);
    b.u2be(1);
    for (int i = 0; i < depth; ++i) {
      b.u1_val('[');
      b.u2be(1);
    }
    b.u1_val('I');
    b.u2be(1);
    RunStripClass(b.bytes());
  }
  printf("PASS: TestAnnotationElementValueInvalidTagAndDeepRecursion\n");
}

void TestMalformedGenericSignatureNoOOB() {
  // Signature attribute with unterminated class type "Lfoo/bar" (no ';')
  ByteBuilder b;
  b.u4be(0xCAFEBABE);
  b.u2be(0);
  b.u2be(61);
  b.u2be(5);
  b.utf8_const("Foo");          // #1
  b.class_const(1);             // #2
  b.utf8_const("Signature");    // #3
  b.utf8_const("<T:Lfoo/bar");  // #4 malformed signature without ';' or '>'
  b.u2be(0x0021);
  b.u2be(2);
  b.u2be(0);
  b.u2be(0);
  b.u2be(0);
  b.u2be(0);
  b.u2be(1);  // 1 class attr
  b.u2be(3);  // Signature
  b.u4be(2);
  b.u2be(4);  // #4
  RunStripClass(b.bytes());
  printf("PASS: TestMalformedGenericSignatureNoOOB\n");
}

class DummyZipProcessor : public ZipExtractorProcessor {
 public:
  bool Accept(const char* /*filename*/, const u4 /*attr*/) override {
    return true;
  }
  void Process(const char* /*filename*/, const u4 /*attr*/, const u1* /*data*/,
               const size_t /*size*/) override {}
};

// Writes bytes to a temporary file and attempts to open/process it with
// ZipExtractor.
void RunZipExtractorOnBytes(const std::vector<u1>& bytes) {
  static int test_counter = 0;
  const char* tmpdir = getenv("TEST_TMPDIR");
  std::string path;
  if (tmpdir != nullptr && tmpdir[0] != '\0') {
    path = tmpdir;
  } else {
    path = ".";
  }
  path += "/ijar_malformed_zip_test_" + std::to_string(++test_counter) + ".zip";

  FILE* f = fopen(path.c_str(), "wb");
  IJAR_TEST_CHECK(f != nullptr);
  if (!bytes.empty()) {
    size_t written = fwrite(bytes.data(), 1, bytes.size(), f);
    IJAR_TEST_CHECK(written == bytes.size());
  }
  fclose(f);

  DummyZipProcessor proc;
  ZipExtractor* ext = ZipExtractor::Create(path.c_str(), &proc);
  if (ext != nullptr) {
    ext->ProcessAll();
    ext->CalculateOutputLength();
    delete ext;
  }
  remove(path.c_str());
}

// Regression test for PR #30995: truncated central directory entry & comment
// OOB.
void TestZipCentralDirTruncatedAndCommentOverflow() {
  // Case 1: file smaller than EOCD (22 bytes)
  RunZipExtractorOnBytes({0x50, 0x4b, 0x05, 0x06});

  // Case 2: EOCD pointing to a central dir entry whose file_comment_length =
  // 0xFFFF past EOF
  {
    ByteBuilder b;
    // Central directory entry at offset 0 (46 bytes header)
    b.u4le(0x02014b50);  // CENTRAL_FILE_HEADER_SIGNATURE
    for (int i = 0; i < 16; ++i) b.u1_val(0);
    b.u4le(0);       // compressed_size
    b.u4le(0);       // uncompressed_size
    b.u2le(4);       // file_name_length = 4 ("a.cl")
    b.u2le(0);       // extra_field_length = 0
    b.u2le(0xFFFF);  // file_comment_length = 65535 (way past EOF!)
    b.u4le(0);
    b.u4le(0);  // attr
    b.u4le(0);  // offset
    b.u1_val('a');
    b.u1_val('.');
    b.u1_val('c');
    b.u1_val('l');

    u4 cd_size = static_cast<u4>(b.bytes().size());
    // EOCD at offset cd_size
    b.u4le(0x06054b50);  // EOCD_SIGNATURE
    b.u2le(0);           // disk
    b.u2le(0);           // disk with cd
    b.u2le(1);           // entries on disk
    b.u2le(1);           // total entries
    b.u4le(cd_size);     // central_dir_size
    b.u4le(0);           // central_dir_offset
    b.u2le(0);           // comment length
    RunZipExtractorOnBytes(b.bytes());
  }
  printf("PASS: TestZipCentralDirTruncatedAndCommentOverflow\n");
}

void TestZipCentralDirOffsetOutOfBoundsAndUnderflow() {
  // Case 1: EOCD with central_dir_offset larger than cand_central_dir - bytes
  // Previously caused unsigned underflow in in_offset_ computation.
  {
    ByteBuilder b;
    for (int i = 0; i < 64; ++i) b.u1_val(0);
    b.u4le(0x06054b50);  // EOCD_SIGNATURE
    b.u2le(0);
    b.u2le(0);
    b.u2le(1);
    b.u2le(1);
    b.u4le(10);   // central_dir_size = 10 (cand_central_dir = 64 - 10 = 54)
    b.u4le(200);  // central_dir_offset = 200 (> 54!)
    b.u2le(0);
    RunZipExtractorOnBytes(b.bytes());
  }
  // Case 2: Central directory entry with local header offset pointing past EOF
  {
    ByteBuilder b;
    b.u4le(0x02014b50);  // CENTRAL_FILE_HEADER_SIGNATURE
    for (int i = 0; i < 16; ++i) b.u1_val(0);
    b.u4le(0);
    b.u4le(0);
    b.u2le(1);  // file_name_length = 1
    b.u2le(0);  // extra_field_length = 0
    b.u2le(0);  // file_comment_length = 0
    b.u4le(0);
    b.u4le(0);
    b.u4le(0x7FFFFFFF);  // local file header offset way past EOF!
    b.u1_val('a');

    u4 cd_size = static_cast<u4>(b.bytes().size());
    b.u4le(0x06054b50);  // EOCD_SIGNATURE
    b.u2le(0);
    b.u2le(0);
    b.u2le(1);
    b.u2le(1);
    b.u4le(cd_size);
    b.u4le(0);
    b.u2le(0);
    RunZipExtractorOnBytes(b.bytes());
  }
  printf("PASS: TestZipCentralDirOffsetOutOfBoundsAndUnderflow\n");
}

void TestZip64ExtraFieldOvershootAndLocatorOOB() {
  // Case 1: Central directory entry with extra_field_length = 6, containing a
  // subheader with data_size = 500 (overshoots extra_end; previously caused
  // infinite/OOB loop).
  {
    ByteBuilder b;
    b.u4le(0x02014b50);  // CENTRAL_FILE_HEADER_SIGNATURE
    for (int i = 0; i < 16; ++i) b.u1_val(0);
    b.u4le(0xFFFFFFFFUL);
    b.u4le(0xFFFFFFFFUL);
    b.u2le(1);  // file_name_length = 1
    b.u2le(6);  // extra_field_length = 6
    b.u2le(0);  // file_comment_length = 0
    b.u4le(0);
    b.u4le(0);
    b.u4le(0);
    b.u1_val('a');
    // Extra field (6 bytes):
    b.u2le(0x0001);  // ZIP64_EXTRA_FIELD_TAG
    b.u2le(500);     // data_size = 500 (> remaining 2 bytes in extra field!)
    b.u2le(0);

    u4 cd_size = static_cast<u4>(b.bytes().size());
    b.u4le(0x06054b50);  // EOCD_SIGNATURE
    b.u2le(0);
    b.u2le(0);
    b.u2le(1);
    b.u2le(1);
    b.u4le(cd_size);
    b.u4le(0);
    b.u2le(0);
    RunZipExtractorOnBytes(b.bytes());
  }

  // Case 2: Zip64 locator pointing out of bounds.
  // Buffer has >= 76 bytes before EOCD and ends with a 20-byte Zip64 locator.
  {
    ByteBuilder b;
    // Pad to >= 56 bytes before the locator
    for (int i = 0; i < 60; ++i) {
      b.u1_val(0);
    }
    // Zip64 locator (20 bytes):
    b.u4le(0x07064b50);  // ZIP64_EOCD_LOCATOR_SIGNATURE
    b.u4le(0);           // disk with zip64 central directory
    b.u8le(0xFFFFFFFFFFFFULL);  // zip64 EOCD offset > in_length!
    b.u4le(1);           // total disks

    // Regular EOCD (22 bytes):
    b.u4le(0x06054b50);  // EOCD_SIGNATURE
    b.u2le(0);
    b.u2le(0);
    b.u2le(0);
    b.u2le(0);
    b.u4le(0);
    b.u4le(0);
    b.u2le(0);
    RunZipExtractorOnBytes(b.bytes());
  }
  printf("PASS: TestZip64ExtraFieldOvershootAndLocatorOOB\n");
}

}  // namespace
}  // namespace devtools_ijar

int main() {
  devtools_ijar::TestTruncatedMagicAndHeader();
  devtools_ijar::TestTruncatedConstantPoolAndUtf8Overflow();
  devtools_ijar::TestConstantPoolDeadSlotAndWrongTag();
  devtools_ijar::TestConstantPoolCycleNoStackOverflow();
  devtools_ijar::TestHeaderAndMemberCountOverflows();
  devtools_ijar::TestAttributeLengthBoundsAndSlice();
  devtools_ijar::TestAnnotationElementValueInvalidTagAndDeepRecursion();
  devtools_ijar::TestMalformedGenericSignatureNoOOB();
  devtools_ijar::TestZipCentralDirTruncatedAndCommentOverflow();
  devtools_ijar::TestZipCentralDirOffsetOutOfBoundsAndUnderflow();
  devtools_ijar::TestZip64ExtraFieldOvershootAndLocatorOOB();
  printf("ALL IJAR MALFORMED TESTS PASSED\n");
  return 0;
}
