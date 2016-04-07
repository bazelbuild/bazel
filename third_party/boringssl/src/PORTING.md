# Porting from OpenSSL to BoringSSL

BoringSSL is an OpenSSL derivative and is mostly source-compatible, for the
subset of OpenSSL retained. Libraries ideally need little to no changes for
BoringSSL support, provided they do not use removed APIs. In general, see if the
library compiles and, on failure, consult the documentation in the header files
and see if problematic features can be removed.

In some cases, BoringSSL-specific code may be necessary. In that case, the
`OPENSSL_IS_BORINGSSL` preprocessor macro may be used in `#ifdef`s. This macro
should also be used in lieu of the presence of any particular function to detect
OpenSSL vs BoringSSL in configure scripts, etc., where those are necessary.

For convenience, BoringSSL defines upstream's `OPENSSL_NO_*` feature macros
corresponding to removed features. These may also be used to disable code which
uses a removed feature.

Note: BoringSSL does *not* have a stable API or ABI. It must be updated with its
consumers. It is not suitable for, say, a system library in a traditional Linux
distribution. For instance, Chromium statically links the specific revision of
BoringSSL it was built against. Likewise, Android's system-internal copy of
BoringSSL is not exposed by the NDK and must not be used by third-party
applications.


## Major API changes

### Integer types

Some APIs have been converted to use `size_t` for consistency and to avoid
integer overflows at the API boundary. (Existing logic uses a mismash of `int`,
`long`, and `unsigned`.)  For the most part, implicit casts mean that existing
code continues to compile. In some cases, this may require BoringSSL-specific
code, particularly to avoid compiler warnings.

Most notably, the `STACK_OF(T)` types have all been converted to use `size_t`
instead of `int` for indices and lengths.

### Reference counts

Some external consumers increment reference counts directly by calling
`CRYPTO_add` with the corresponding `CRYPTO_LOCK_*` value.

These APIs no longer exist in BoringSSL. Instead, code which increments
reference counts should call the corresponding `FOO_up_ref` function, such as
`EVP_PKEY_up_ref`. Note that not all of these APIs are present in OpenSSL and
may require `#ifdef`s.

### Error codes

OpenSSL's errors are extremely specific, leaking internals of the library,
including even a function code for the function which emitted the error! As some
logic in BoringSSL has been rewritten, code which conditions on the error may
break (grep for `ERR_GET_REASON` and `ERR_GET_FUNC`). This danger also exists
when upgrading OpenSSL versions.

Where possible, avoid conditioning on the exact error reason. Otherwise, a
BoringSSL `#ifdef` may be necessary. Exactly how best to resolve this issue is
still being determined. It's possible some new APIs will be added in the future.

Function codes have been completely removed. Remove code which conditions on
these as it will break with the slightest change in the library, OpenSSL or
BoringSSL.

### `*_ctrl` functions

Some OpenSSL APIs are implemented with `ioctl`-style functions such as
`SSL_ctrl` and `EVP_PKEY_CTX_ctrl`, combined with convenience macros, such as

    # define SSL_CTX_set_mode(ctx,op) \
            SSL_CTX_ctrl((ctx),SSL_CTRL_MODE,(op),NULL)

In BoringSSL, these macros have been replaced with proper functions. The
underlying `_ctrl` functions have been removed.

For convenience, `SSL_CTRL_*` values are retained as macros to `doesnt_exist` so
existing code which uses them (or the wrapper macros) in `#ifdef` expressions
will continue to function. However, the macros themselves will not work.

Switch any `*_ctrl` callers to the macro/function versions. This works in both
OpenSSL and BoringSSL. Note that BoringSSL's function versions will be
type-checked and may require more care with types.

### HMAC `EVP_PKEY`s

`EVP_PKEY_HMAC` is removed. Use the `HMAC_*` functions in `hmac.h` instead. This
is compatible with OpenSSL.

### DSA `EVP_PKEY`s

`EVP_PKEY_DSA` is deprecated. It is currently still possible to parse DER into a
DSA `EVP_PKEY`, but signing or verifying with those objects will not work.

### DES

The `DES_cblock` type has been switched from an array to a struct to avoid the
pitfalls around array types in C. Where features which require DES cannot be
disabled, BoringSSL-specific codepaths may be necessary.

### TLS renegotiation

OpenSSL enables TLS renegotiation by default and accepts renegotiation requests
from the peer transparently. Renegotiation is an extremely problematic protocol
feature, so BoringSSL rejects peer renegotiations by default.

To enable renegotiation, call `SSL_set_reject_peer_renegotiations` and set it to
off. Renegotiation is only supported as a client in SSL3/TLS and the
HelloRequest must be received at a quiet point in the application protocol. This
is sufficient to support the common use of requesting a new client certificate
between an HTTP request and response in (unpipelined) HTTP/1.1.

Things which do not work:

* There is no support for renegotiation as a server.

* There is no support for renegotiation in DTLS.

* There is no support for initiating renegotiation; `SSL_renegotiate` always
  fails and `SSL_set_state` does nothing.

* Interleaving application data with the new handshake is forbidden.

* If a HelloRequest is received while `SSL_write` has unsent application data,
  the renegotiation is rejected.

### Lowercase hexadecimal

BoringSSL's `BN_bn2hex` function uses lowercase hexadecimal digits instead of
uppercase. Some code may require changes to avoid being sensitive to this
difference.


## Optional BoringSSL-specific simplifications

BoringSSL makes some changes to OpenSSL which simplify the API but remain
compatible with OpenSSL consumers. In general, consult the BoringSSL
documentation for any functions in new BoringSSL-only code.

### Return values

Most OpenSSL APIs return 1 on success and either 0 or -1 on failure. BoringSSL
has narrowed most of these to 1 on success and 0 on failure. BoringSSL-specific
code may take advantage of the less error-prone APIs and use `!` to check for
errors.

### Initialization

OpenSSL has a number of different initialization functions for setting up error
strings and loading algorithms, etc. All of these functions still exist in
BoringSSL for convenience, but they do nothing and are not necessary.

The one exception is `CRYPTO_library_init` (and `SSL_library_init` which merely
calls it). In `BORINGSSL_NO_STATIC_INITIALIZER` builds, it must be called to
query CPU capabitilies before the rest of the library. In the default
configuration, this is done with a static initializer and is also unnecessary.

### Threading

OpenSSL provides a number of APIs to configure threading callbacks and set up
locks. Without initializing these, the library is not thread-safe. Configuring
these does nothing in BoringSSL. Instead, BoringSSL calls pthreads and the
corresponding Windows APIs internally and is always thread-safe where the API
guarantees it.
