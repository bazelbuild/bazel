/* Copyright (c) 2015, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#ifndef OPENSSL_HEADER_CRYPTO_TEST_STL_COMPAT_H
#define OPENSSL_HEADER_CRYPTO_TEST_STL_COMPAT_H

#include <assert.h>

#include <vector>


// This header contains re-implementations of library functions from C++11. They
// will be replaced with their standard counterparts once Chromium has C++11
// library support in its toolchain.

namespace bssl {

// vector_data is a reimplementation of |std::vector::data| from C++11.
template <class T>
static T *vector_data(std::vector<T> *out) {
  return out->empty() ? nullptr : &(*out)[0];
}

template <class T>
static const T *vector_data(const std::vector<T> *out) {
  return out->empty() ? nullptr : &(*out)[0];
}

// remove_reference is a reimplementation of |std::remove_reference| from C++11.
template <class T>
struct remove_reference {
  using type = T;
};

template <class T>
struct remove_reference<T&> {
  using type = T;
};

template <class T>
struct remove_reference<T&&> {
  using type = T;
};

// move is a reimplementation of |std::move| from C++11.
template <class T>
typename remove_reference<T>::type &&move(T &&t) {
  return static_cast<typename remove_reference<T>::type&&>(t);
}

// default_delete is a partial reimplementation of |std::default_delete| from
// C++11.
template <class T>
struct default_delete {
  void operator()(T *t) const {
    enum { type_must_be_complete = sizeof(T) };
    delete t;
  }
};

// nullptr_t is |std::nullptr_t| from C++11.
using nullptr_t = decltype(nullptr);

// unique_ptr is a partial reimplementation of |std::unique_ptr| from C++11. It
// intentionally does not support stateful deleters to avoid having to bother
// with the empty member optimization.
template <class T, class Deleter = default_delete<T>>
class unique_ptr {
 public:
  unique_ptr() : ptr_(nullptr) {}
  unique_ptr(nullptr_t) : ptr_(nullptr) {}
  unique_ptr(T *ptr) : ptr_(ptr) {}
  unique_ptr(const unique_ptr &u) = delete;

  unique_ptr(unique_ptr &&u) : ptr_(nullptr) {
    reset(u.release());
  }

  ~unique_ptr() {
    reset();
  }

  unique_ptr &operator=(nullptr_t) {
    reset();
    return *this;
  }

  unique_ptr &operator=(unique_ptr &&u) {
    reset(u.release());
    return *this;
  }

  unique_ptr& operator=(const unique_ptr &u) = delete;

  explicit operator bool() const {
    return ptr_ != nullptr;
  }

  T &operator*() const {
    assert(ptr_ != nullptr);
    return *ptr_;
  }

  T *operator->() const {
    assert(ptr_ != nullptr);
    return ptr_;
  }

  T *get() const {
    return ptr_;
  }

  T *release() {
    T *ptr = ptr_;
    ptr_ = nullptr;
    return ptr;
  }

  void reset(T *ptr = nullptr) {
    if (ptr_ != nullptr) {
      Deleter()(ptr_);
    }
    ptr_ = ptr;
  }

 private:
  T *ptr_;
};

}  // namespace bssl


#endif  // OPENSSL_HEADER_CRYPTO_TEST_STL_COMPAT_H
