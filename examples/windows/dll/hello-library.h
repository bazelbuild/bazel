#ifndef HELLO_LIBRARY_H
#define HELLO_LIBRARY_H

#if COMPILING_DLL
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT __declspec(dllimport)
#endif

extern "C" DLLEXPORT char *get_time();
extern "C" DLLEXPORT void say_hello(char *);

#endif