// Copyright 2024 The Bazel Authors. All rights reserved.
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

#include <jni.h>

#include "src/main/cpp/util/logging.h"
#include "src/main/native/latin1_jni_path.h"
#include "sqlite3.h"

namespace blaze_jni {

namespace {

// A RAII wrapper around a null-terminated UTF string obtained from a jstring.
class ScopedUTFString {
 public:
  ScopedUTFString(JNIEnv *env, jstring jstr)
      : env_(env),
        jstr_(jstr),
        str_(env->GetStringUTFChars(jstr, nullptr)),
        len_(env->GetStringUTFLength(jstr)) {}

  ~ScopedUTFString() { env_->ReleaseStringUTFChars(jstr_, str_); }

  const char *c_str() const { return str_; }

  int length() const { return len_; }

 private:
  JNIEnv *env_;
  jstring jstr_;
  const char *str_;
  int len_;
};

// Throws an exception for the given SQLite error code.
void PostException(JNIEnv *env, int err) {
  jclass exc_class = env->FindClass(
      "com/google/devtools/build/lib/remote/disk/Sqlite$SqliteException");
  if (exc_class == nullptr) {
    BAZEL_LOG(FATAL) << "Failed to throw SQLite exception from JNI";
  }
  jmethodID exc_ctor = env->GetMethodID(exc_class, "<init>", "(I)V");
  if (exc_ctor == nullptr) {
    BAZEL_LOG(FATAL) << "Failed to throw SQLite exception from JNI";
  }
  jthrowable exc =
      static_cast<jthrowable>(env->NewObject(exc_class, exc_ctor, err));
  if (env->Throw(exc)) {
    BAZEL_LOG(FATAL) << "Failed to throw SQLite exception from JNI";
  }
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_errStr(JNIEnv *env,
                                                             jclass cls,
                                                             jint err) {
  const char *str = sqlite3_errstr(err);
  if (str == nullptr) {
    str = "unknown error";
  }
  return env->NewStringUTF(str);
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_openConn(
    JNIEnv *env, jclass cls, jstring path_str) {
  const char *path = GetStringLatin1Chars(env, path_str);
  sqlite3 *conn = nullptr;
  int err = sqlite3_open(path, &conn);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
  ReleaseStringLatin1Chars(path);
  return reinterpret_cast<jlong>(conn);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_closeConn(
    JNIEnv *env, jclass cls, jlong conn_ptr) {
  sqlite3 *conn = reinterpret_cast<sqlite3 *>(conn_ptr);
  int err = sqlite3_close(conn);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_prepareStmt(
    JNIEnv *env, jclass cls, jlong conn_ptr, jstring sql_jstr) {
  sqlite3 *conn = reinterpret_cast<sqlite3 *>(conn_ptr);
  ScopedUTFString sql(env, sql_jstr);
  sqlite3_stmt *stmt;
  const char *sql_tail;
  int err = sqlite3_prepare_v3(conn, sql.c_str(), sql.length(),
                               SQLITE_PREPARE_PERSISTENT, &stmt, &sql_tail);
  if (err == SQLITE_OK && *sql_tail != '\0') {
    // Return special value to signal an unsupported multi-statement string.
    sqlite3_finalize(stmt);
    return -1L;
  }
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
  return reinterpret_cast<jlong>(stmt);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_bindStmtLong(
    JNIEnv *env, jclass cls, jlong stmt_ptr, jint i, jlong val) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  int err = sqlite3_bind_int64(stmt, i, val);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_bindStmtDouble(
    JNIEnv *env, jclass cls, jlong stmt_ptr, jint i, jdouble val) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  int err = sqlite3_bind_double(stmt, i, val);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_bindStmtString(
    JNIEnv *env, jclass cls, jlong stmt_ptr, jint i, jstring val_jstr) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  ScopedUTFString val(env, val_jstr);
  int err =
      sqlite3_bind_text(stmt, i, val.c_str(), val.length(), SQLITE_TRANSIENT);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_clearStmtBinding(
    JNIEnv *env, jclass cls, jlong stmt_ptr, jint i) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  int err = sqlite3_bind_null(stmt, i);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_clearStmtBindings(
    JNIEnv *env, jclass cls, jlong stmt_ptr) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  int err = sqlite3_clear_bindings(stmt);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_stepStmt(JNIEnv *env,
                                                               jclass cls,
                                                               jlong stmt_ptr) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  int err = sqlite3_step(stmt);
  // Deviation from the C API: SQLITE_ROW and SQLITE_DONE are returned as true
  // or false, respectively. Everything else causes an exception to be thrown.
  if (err == SQLITE_ROW) {
    return true;
  } else if (err == SQLITE_DONE) {
    return false;
  }
  PostException(env, err);
  return false;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_columnIsNull(
    JNIEnv *env, jclass cls, jlong stmt_ptr, jint i) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  return sqlite3_column_type(stmt, i) == SQLITE_NULL;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_columnLong(JNIEnv *env,
                                                                 jclass cls,
                                                                 jlong stmt_ptr,
                                                                 jint i) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  return sqlite3_column_int64(stmt, i);
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_columnDouble(
    JNIEnv *env, jclass cls, jlong stmt_ptr, jint i) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  return sqlite3_column_double(stmt, i);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_columnString(
    JNIEnv *env, jclass cls, jlong stmt_ptr, jint i) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  const char *val =
      reinterpret_cast<const char *>(sqlite3_column_text(stmt, i));
  return val != nullptr ? env->NewStringUTF(val) : env->NewStringUTF("");
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_resetStmt(
    JNIEnv *env, jclass cls, jlong stmt_ptr) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  int err = sqlite3_reset(stmt);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_devtools_build_lib_remote_disk_Sqlite_finalizeStmt(
    JNIEnv *env, jclass cls, jlong stmt_ptr) {
  sqlite3_stmt *stmt = reinterpret_cast<sqlite3_stmt *>(stmt_ptr);
  int err = sqlite3_finalize(stmt);
  if (err != SQLITE_OK) {
    PostException(env, err);
  }
}

}  // namespace blaze_jni
