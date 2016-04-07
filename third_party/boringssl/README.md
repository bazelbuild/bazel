Update instructions:

1. `git clone https://boringssl.googlesource.com/boringssl`
2. `git checkout <commithash>` (currently `82aa28fa`)
3. Build BoringSSL according to its instructions
4. `mkdir -p third_party/boringssl/src`
5. `cp -R <BoringSSL source tree>/* third_party/boringssl/src`
6. `cp <BoringSSL source tree>/build/crypto/err/err_data.c third_party/boringssl`
7. Done.
