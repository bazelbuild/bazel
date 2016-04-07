cmake_minimum_required (VERSION 2.8.10)

project (BoringSSL)

if(ANDROID)
  # Android-NDK CMake files reconfigure the path and so Go and Perl won't be
  # found. However, ninja will still find them in $PATH if we just name them.
  set(PERL_EXECUTABLE "perl")
  set(GO_EXECUTABLE "go")
else()
  find_package(Perl REQUIRED)
  find_program(GO_EXECUTABLE go)
endif()

if (NOT GO_EXECUTABLE)
  message(FATAL_ERROR "Could not find Go")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -ggdb -fvisibility=hidden")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -ggdb -std=c++0x -fvisibility=hidden")
elseif(MSVC)
  set(MSVC_DISABLED_WARNINGS_LIST
      "C4100" # 'exarg' : unreferenced formal parameter
      "C4127" # conditional expression is constant
      "C4200" # nonstandard extension used : zero-sized array in
              # struct/union.
      "C4210" # nonstandard extension used : function given file scope
      "C4242" # 'function' : conversion from 'int' to 'uint8_t',
              # possible loss of data
      "C4244" # 'function' : conversion from 'int' to 'uint8_t',
              # possible loss of data
      "C4245" # 'initializing' : conversion from 'long' to
              # 'unsigned long', signed/unsigned mismatch
      "C4267" # conversion from 'size_t' to 'int', possible loss of data
      "C4371" # layout of class may have changed from a previous version of the
              # compiler due to better packing of member '...'
      "C4388" # signed/unsigned mismatch
      "C4296" # '>=' : expression is always true
      "C4350" # behavior change: 'std::_Wrap_alloc...'
      "C4365" # '=' : conversion from 'size_t' to 'int',
              # signed/unsigned mismatch
      "C4389" # '!=' : signed/unsigned mismatch
      "C4510" # 'argument' : default constructor could not be generated
      "C4512" # 'argument' : assignment operator could not be generated
      "C4514" # 'function': unreferenced inline function has been removed
      "C4548" # expression before comma has no effect; expected expression with
              # side-effect" caused by FD_* macros.
      "C4610" # struct 'argument' can never be instantiated - user defined
              # constructor required.
      "C4625" # copy constructor could not be generated because a base class
              # copy constructor is inaccessible or deleted
      "C4626" # assignment operator could not be generated because a base class
              # assignment operator is inaccessible or deleted
      "C4706" # assignment within conditional expression
      "C4710" # 'function': function not inlined
      "C4711" # function 'function' selected for inline expansion
      "C4800" # 'int' : forcing value to bool 'true' or 'false'
              # (performance warning)
      "C4820" # 'bytes' bytes padding added after construct 'member_name'
      "C4996" # 'read': The POSIX name for this item is deprecated. Instead,
              # use the ISO C++ conformant name: _read.
     )
  string(REPLACE "C" " -wd" MSVC_DISABLED_WARNINGS_STR
                            ${MSVC_DISABLED_WARNINGS_LIST})
  set(CMAKE_C_FLAGS   "-Wall -WX ${MSVC_DISABLED_WARNINGS_STR}")
  set(CMAKE_CXX_FLAGS "-Wall -WX ${MSVC_DISABLED_WARNINGS_STR}")
  add_definitions(-D_HAS_EXCEPTIONS=0)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DNOMINMAX)
endif()

if((CMAKE_COMPILER_IS_GNUCXX AND CMAKE_C_COMPILER_VERSION VERSION_GREATER "4.7.99") OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wshadow")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
endif()

if((CMAKE_COMPILER_IS_GNUCXX AND CMAKE_C_COMPILER_VERSION VERSION_GREATER "4.8.99") OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -D_XOPEN_SOURCE=700")
endif()

add_definitions(-DBORINGSSL_IMPLEMENTATION)

if (BUILD_SHARED_LIBS)
  add_definitions(-DBORINGSSL_SHARED_LIBRARY)
  # Enable position-independent code globally. This is needed because
  # some library targets are OBJECT libraries.
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
endif()

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(ARCH "x86_64")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "amd64")
  set(ARCH "x86_64")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
  # cmake reports AMD64 on Windows, but we might be building for 32-bit.
  if (CMAKE_CL_64)
    set(ARCH "x86_64")
  else()
    set(ARCH "x86")
  endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86")
  set(ARCH "x86")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i386")
  set(ARCH "x86")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i686")
  set(ARCH "x86")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm")
  set(ARCH "arm")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv6")
  set(ARCH "arm")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7-a")
  set(ARCH "arm")
elseif (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(ARCH "aarch64")
else()
  message(FATAL_ERROR "Unknown processor:" ${CMAKE_SYSTEM_PROCESSOR})
endif()

if (ANDROID AND ${ARCH} STREQUAL "arm")
  # The Android-NDK CMake files somehow fail to set the -march flag for
  # assembly files. Without this flag, the compiler believes that it's
  # building for ARMv5.
  set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -march=${CMAKE_SYSTEM_PROCESSOR}")
endif()

if (${ARCH} STREQUAL "x86" AND APPLE)
  # With CMake 2.8.x, ${CMAKE_SYSTEM_PROCESSOR} evalutes to i386 on OS X,
  # but clang defaults to 64-bit builds on OS X unless otherwise told.
  # Set ARCH to x86_64 so clang and CMake agree. This is fixed in CMake 3.
  set(ARCH "x86_64")
endif()

if (OPENSSL_NO_ASM)
  add_definitions(-DOPENSSL_NO_ASM)
  set(ARCH "generic")
endif()

add_subdirectory(crypto)
add_subdirectory(ssl)
add_subdirectory(ssl/test)
add_subdirectory(tool)
add_subdirectory(decrepit)
