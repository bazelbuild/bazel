#define PY_SSIZE_T_CLEAN 1
#include <Python.h>
#include <bytesobject.h>
#include <structmember.h>
#include <vector>
#include "../common/version.h"
#include <brotli/decode.h>
#include <brotli/encode.h>

#if PY_MAJOR_VERSION >= 3
#define PyInt_Check PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#endif

static PyObject *BrotliError;

static int as_bounded_int(PyObject *o, int* result, int lower_bound, int upper_bound) {
  long value = PyInt_AsLong(o);
  if ((value < (long) lower_bound) || (value > (long) upper_bound)) {
    return 0;
  }
  *result = (int) value;
  return 1;
}

static int mode_convertor(PyObject *o, BrotliEncoderMode *mode) {
  if (!PyInt_Check(o)) {
    PyErr_SetString(BrotliError, "Invalid mode");
    return 0;
  }

  int mode_value = -1;
  if (!as_bounded_int(o, &mode_value, 0, 255)) {
    PyErr_SetString(BrotliError, "Invalid mode");
    return 0;
  }
  *mode = (BrotliEncoderMode) mode_value;
  if (*mode != BROTLI_MODE_GENERIC &&
      *mode != BROTLI_MODE_TEXT &&
      *mode != BROTLI_MODE_FONT) {
    PyErr_SetString(BrotliError, "Invalid mode");
    return 0;
  }

  return 1;
}

static int quality_convertor(PyObject *o, int *quality) {
  if (!PyInt_Check(o)) {
    PyErr_SetString(BrotliError, "Invalid quality");
    return 0;
  }

  if (!as_bounded_int(o, quality, 0, 11)) {
    PyErr_SetString(BrotliError, "Invalid quality. Range is 0 to 11.");
    return 0;
  }

  return 1;
}

static int lgwin_convertor(PyObject *o, int *lgwin) {
  if (!PyInt_Check(o)) {
    PyErr_SetString(BrotliError, "Invalid lgwin");
    return 0;
  }

  if (!as_bounded_int(o, lgwin, 10, 24)) {
    PyErr_SetString(BrotliError, "Invalid lgwin. Range is 10 to 24.");
    return 0;
  }

  return 1;
}

static int lgblock_convertor(PyObject *o, int *lgblock) {
  if (!PyInt_Check(o)) {
    PyErr_SetString(BrotliError, "Invalid lgblock");
    return 0;
  }

  if (!as_bounded_int(o, lgblock, 0, 24) || (*lgblock != 0 && *lgblock < 16)) {
    PyErr_SetString(BrotliError, "Invalid lgblock. Can be 0 or in range 16 to 24.");
    return 0;
  }

  return 1;
}

static BROTLI_BOOL compress_stream(BrotliEncoderState* enc, BrotliEncoderOperation op,
                                   std::vector<uint8_t>* output,
                                   uint8_t* input, size_t input_length) {
  BROTLI_BOOL ok = BROTLI_TRUE;
  Py_BEGIN_ALLOW_THREADS

  size_t available_in = input_length;
  const uint8_t* next_in = input;
  size_t available_out = 0;
  uint8_t* next_out = NULL;

  while (ok) {
    ok = BrotliEncoderCompressStream(enc, op,
                                     &available_in, &next_in,
                                     &available_out, &next_out, NULL);
    if (!ok)
      break;

    size_t buffer_length = 0; // Request all available output.
    const uint8_t* buffer = BrotliEncoderTakeOutput(enc, &buffer_length);
    if (buffer_length) {
      (*output).insert((*output).end(), buffer, buffer + buffer_length);
    }

    if (available_in || BrotliEncoderHasMoreOutput(enc)) {
      continue;
    }

    break;
  }

  Py_END_ALLOW_THREADS
  return ok;
}

PyDoc_STRVAR(brotli_Compressor_doc,
"An object to compress a byte string.\n"
"\n"
"Signature:\n"
"  Compressor(mode=MODE_GENERIC, quality=11, lgwin=22, lgblock=0)\n"
"\n"
"Args:\n"
"  mode (int, optional): The compression mode can be MODE_GENERIC (default),\n"
"    MODE_TEXT (for UTF-8 format text input) or MODE_FONT (for WOFF 2.0). \n"
"  quality (int, optional): Controls the compression-speed vs compression-\n"
"    density tradeoff. The higher the quality, the slower the compression.\n"
"    Range is 0 to 11. Defaults to 11.\n"
"  lgwin (int, optional): Base 2 logarithm of the sliding window size. Range\n"
"    is 10 to 24. Defaults to 22.\n"
"  lgblock (int, optional): Base 2 logarithm of the maximum input block size.\n"
"    Range is 16 to 24. If set to 0, the value will be set based on the\n"
"    quality. Defaults to 0.\n"
"\n"
"Raises:\n"
"  brotli.error: If arguments are invalid.\n");

typedef struct {
  PyObject_HEAD
  BrotliEncoderState* enc;
} brotli_Compressor;

static void brotli_Compressor_dealloc(brotli_Compressor* self) {
  BrotliEncoderDestroyInstance(self->enc);
  #if PY_MAJOR_VERSION >= 3
  Py_TYPE(self)->tp_free((PyObject*)self);
  #else
  self->ob_type->tp_free((PyObject*)self);
  #endif
}

static PyObject* brotli_Compressor_new(PyTypeObject *type, PyObject *args, PyObject *keywds) {
  brotli_Compressor *self;
  self = (brotli_Compressor *)type->tp_alloc(type, 0);

  if (self != NULL) {
    self->enc = BrotliEncoderCreateInstance(0, 0, 0);
  }

  return (PyObject *)self;
}

static int brotli_Compressor_init(brotli_Compressor *self, PyObject *args, PyObject *keywds) {
  BrotliEncoderMode mode = (BrotliEncoderMode) -1;
  int quality = -1;
  int lgwin = -1;
  int lgblock = -1;
  int ok;

  static const char *kwlist[] = {"mode", "quality", "lgwin", "lgblock", NULL};

  ok = PyArg_ParseTupleAndKeywords(args, keywds, "|O&O&O&O&:Compressor",
                    const_cast<char **>(kwlist),
                    &mode_convertor, &mode,
                    &quality_convertor, &quality,
                    &lgwin_convertor, &lgwin,
                    &lgblock_convertor, &lgblock);
  if (!ok)
    return -1;
  if (!self->enc)
    return -1;

  if ((int) mode != -1)
    BrotliEncoderSetParameter(self->enc, BROTLI_PARAM_MODE, (uint32_t)mode);
  if (quality != -1)
    BrotliEncoderSetParameter(self->enc, BROTLI_PARAM_QUALITY, (uint32_t)quality);
  if (lgwin != -1)
    BrotliEncoderSetParameter(self->enc, BROTLI_PARAM_LGWIN, (uint32_t)lgwin);
  if (lgblock != -1)
    BrotliEncoderSetParameter(self->enc, BROTLI_PARAM_LGBLOCK, (uint32_t)lgblock);

  return 0;
}

PyDoc_STRVAR(brotli_Compressor_process_doc,
"Process \"string\" for compression, returning a string that contains \n"
"compressed output data.  This data should be concatenated to the output \n"
"produced by any preceding calls to the \"process()\" or flush()\" methods. \n"
"Some or all of the input may be kept in internal buffers for later \n"
"processing, and the compressed output data may be empty until enough input \n"
"has been accumulated.\n"
"\n"
"Signature:\n"
"  compress(string)\n"
"\n"
"Args:\n"
"  string (bytes): The input data\n"
"\n"
"Returns:\n"
"  The compressed output data (bytes)\n"
"\n"
"Raises:\n"
"  brotli.error: If compression fails\n");

static PyObject* brotli_Compressor_process(brotli_Compressor *self, PyObject *args) {
  PyObject* ret = NULL;
  std::vector<uint8_t> output;
  Py_buffer input;
  BROTLI_BOOL ok = BROTLI_TRUE;

#if PY_MAJOR_VERSION >= 3
  ok = (BROTLI_BOOL)PyArg_ParseTuple(args, "y*:process", &input);
#else
  ok = (BROTLI_BOOL)PyArg_ParseTuple(args, "s*:process", &input);
#endif

  if (!ok)
    return NULL;

  if (!self->enc) {
    ok = BROTLI_FALSE;
    goto end;
  }

  ok = compress_stream(self->enc, BROTLI_OPERATION_PROCESS,
                       &output, static_cast<uint8_t*>(input.buf), input.len);

end:
  PyBuffer_Release(&input);
  if (ok) {
    ret = PyBytes_FromStringAndSize((char*)(output.size() ? &output[0] : NULL), output.size());
  } else {
    PyErr_SetString(BrotliError, "BrotliEncoderCompressStream failed while processing the stream");
  }

  return ret;
}

PyDoc_STRVAR(brotli_Compressor_flush_doc,
"Process all pending input, returning a string containing the remaining\n"
"compressed data. This data should be concatenated to the output produced by\n"
"any preceding calls to the \"process()\" or \"flush()\" methods.\n"
"\n"
"Signature:\n"
"  flush()\n"
"\n"
"Returns:\n"
"  The compressed output data (bytes)\n"
"\n"
"Raises:\n"
"  brotli.error: If compression fails\n");

static PyObject* brotli_Compressor_flush(brotli_Compressor *self) {
  PyObject *ret = NULL;
  std::vector<uint8_t> output;
  BROTLI_BOOL ok = BROTLI_TRUE;

  if (!self->enc) {
    ok = BROTLI_FALSE;
    goto end;
  }

  ok = compress_stream(self->enc, BROTLI_OPERATION_FLUSH,
                       &output, NULL, 0);

end:
  if (ok) {
    ret = PyBytes_FromStringAndSize((char*)(output.size() ? &output[0] : NULL), output.size());
  } else {
    PyErr_SetString(BrotliError, "BrotliEncoderCompressStream failed while flushing the stream");
  }

  return ret;
}

PyDoc_STRVAR(brotli_Compressor_finish_doc,
"Process all pending input and complete all compression, returning a string\n"
"containing the remaining compressed data. This data should be concatenated\n"
"to the output produced by any preceding calls to the \"process()\" or\n"
"\"flush()\" methods.\n"
"After calling \"finish()\", the \"process()\" and \"flush()\" methods\n"
"cannot be called again, and a new \"Compressor\" object should be created.\n"
"\n"
"Signature:\n"
"  finish(string)\n"
"\n"
"Returns:\n"
"  The compressed output data (bytes)\n"
"\n"
"Raises:\n"
"  brotli.error: If compression fails\n");

static PyObject* brotli_Compressor_finish(brotli_Compressor *self) {
  PyObject *ret = NULL;
  std::vector<uint8_t> output;
  BROTLI_BOOL ok = BROTLI_TRUE;

  if (!self->enc) {
    ok = BROTLI_FALSE;
    goto end;
  }

  ok = compress_stream(self->enc, BROTLI_OPERATION_FINISH,
                       &output, NULL, 0);

  if (ok) {
    ok = BrotliEncoderIsFinished(self->enc);
  }

end:
  if (ok) {
    ret = PyBytes_FromStringAndSize((char*)(output.empty() ? NULL : &output[0]), output.size());
  } else {
    PyErr_SetString(BrotliError, "BrotliEncoderCompressStream failed while finishing the stream");
  }

  return ret;
}

static PyMemberDef brotli_Compressor_members[] = {
  {NULL}  /* Sentinel */
};

static PyMethodDef brotli_Compressor_methods[] = {
  {"process", (PyCFunction)brotli_Compressor_process, METH_VARARGS, brotli_Compressor_process_doc},
  {"flush", (PyCFunction)brotli_Compressor_flush, METH_NOARGS, brotli_Compressor_flush_doc},
  {"finish", (PyCFunction)brotli_Compressor_finish, METH_NOARGS, brotli_Compressor_finish_doc},
  {NULL}  /* Sentinel */
};

static PyTypeObject brotli_CompressorType = {
  #if PY_MAJOR_VERSION >= 3
  PyVarObject_HEAD_INIT(NULL, 0)
  #else
  PyObject_HEAD_INIT(NULL)
  0,                                     /* ob_size*/
  #endif
  "brotli.Compressor",                   /* tp_name */
  sizeof(brotli_Compressor),             /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)brotli_Compressor_dealloc, /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_compare */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  brotli_Compressor_doc,                 /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  brotli_Compressor_methods,             /* tp_methods */
  brotli_Compressor_members,             /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  (initproc)brotli_Compressor_init,      /* tp_init */
  0,                                     /* tp_alloc */
  brotli_Compressor_new,                 /* tp_new */
};

static BROTLI_BOOL decompress_stream(BrotliDecoderState* dec,
                                     std::vector<uint8_t>* output,
                                     uint8_t* input, size_t input_length) {
  BROTLI_BOOL ok = BROTLI_TRUE;
  Py_BEGIN_ALLOW_THREADS

  size_t available_in = input_length;
  const uint8_t* next_in = input;
  size_t available_out = 0;
  uint8_t* next_out = NULL;

  BrotliDecoderResult result = BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT;
  while (result == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
    result = BrotliDecoderDecompressStream(dec,
                                           &available_in, &next_in,
                                           &available_out, &next_out, NULL);
    size_t buffer_length = 0; // Request all available output.
    const uint8_t* buffer = BrotliDecoderTakeOutput(dec, &buffer_length);
    if (buffer_length) {
      (*output).insert((*output).end(), buffer, buffer + buffer_length);
    }
  }
  ok = result != BROTLI_DECODER_RESULT_ERROR && !available_in;

  Py_END_ALLOW_THREADS
  return ok;
}

PyDoc_STRVAR(brotli_Decompressor_doc,
"An object to decompress a byte string.\n"
"\n"
"Signature:\n"
"  Decompressor()\n"
"\n"
"Raises:\n"
"  brotli.error: If arguments are invalid.\n");

typedef struct {
  PyObject_HEAD
  BrotliDecoderState* dec;
} brotli_Decompressor;

static void brotli_Decompressor_dealloc(brotli_Decompressor* self) {
  BrotliDecoderDestroyInstance(self->dec);
  #if PY_MAJOR_VERSION >= 3
  Py_TYPE(self)->tp_free((PyObject*)self);
  #else
  self->ob_type->tp_free((PyObject*)self);
  #endif
}

static PyObject* brotli_Decompressor_new(PyTypeObject *type, PyObject *args, PyObject *keywds) {
  brotli_Decompressor *self;
  self = (brotli_Decompressor *)type->tp_alloc(type, 0);

  if (self != NULL) {
    self->dec = BrotliDecoderCreateInstance(0, 0, 0);
  }

  return (PyObject *)self;
}

static int brotli_Decompressor_init(brotli_Decompressor *self, PyObject *args, PyObject *keywds) {
  int ok;

  static const char *kwlist[] = {NULL};

  ok = PyArg_ParseTupleAndKeywords(args, keywds, "|:Decompressor",
                    const_cast<char **>(kwlist));
  if (!ok)
    return -1;
  if (!self->dec)
    return -1;

  return 0;
}

PyDoc_STRVAR(brotli_Decompressor_process_doc,
"Process \"string\" for decompression, returning a string that contains \n"
"decompressed output data.  This data should be concatenated to the output \n"
"produced by any preceding calls to the \"process()\" method. \n"
"Some or all of the input may be kept in internal buffers for later \n"
"processing, and the decompressed output data may be empty until enough input \n"
"has been accumulated.\n"
"\n"
"Signature:\n"
"  decompress(string)\n"
"\n"
"Args:\n"
"  string (bytes): The input data\n"
"\n"
"Returns:\n"
"  The decompressed output data (bytes)\n"
"\n"
"Raises:\n"
"  brotli.error: If decompression fails\n");

static PyObject* brotli_Decompressor_process(brotli_Decompressor *self, PyObject *args) {
  PyObject* ret = NULL;
  std::vector<uint8_t> output;
  Py_buffer input;
  BROTLI_BOOL ok = BROTLI_TRUE;

#if PY_MAJOR_VERSION >= 3
  ok = (BROTLI_BOOL)PyArg_ParseTuple(args, "y*:process", &input);
#else
  ok = (BROTLI_BOOL)PyArg_ParseTuple(args, "s*:process", &input);
#endif

  if (!ok)
    return NULL;

  if (!self->dec) {
    ok = BROTLI_FALSE;
    goto end;
  }

  ok = decompress_stream(self->dec, &output, static_cast<uint8_t*>(input.buf), input.len);

end:
  PyBuffer_Release(&input);
  if (ok) {
    ret = PyBytes_FromStringAndSize((char*)(output.empty() ? NULL : &output[0]), output.size());
  } else {
    PyErr_SetString(BrotliError, "BrotliDecoderDecompressStream failed while processing the stream");
  }

  return ret;
}

PyDoc_STRVAR(brotli_Decompressor_is_finished_doc,
"Checks if decoder instance reached the final state.\n"
"\n"
"Signature:\n"
"  is_finished()\n"
"\n"
"Returns:\n"
"  True  if the decoder is in a state where it reached the end of the input\n"
"        and produced all of the output\n"
"  False otherwise\n"
"\n"
"Raises:\n"
"  brotli.error: If decompression fails\n");

static PyObject* brotli_Decompressor_is_finished(brotli_Decompressor *self) {
  PyObject *ret = NULL;
  std::vector<uint8_t> output;
  BROTLI_BOOL ok = BROTLI_TRUE;

  if (!self->dec) {
    ok = BROTLI_FALSE;
    PyErr_SetString(BrotliError, "BrotliDecoderState is NULL while checking is_finished");
    goto end;
  }

  if (BrotliDecoderIsFinished(self->dec)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }

end:
  if (ok) {
    ret = PyBytes_FromStringAndSize((char*)(output.empty() ? NULL : &output[0]), output.size());
  } else {
    PyErr_SetString(BrotliError, "BrotliDecoderDecompressStream failed while finishing the stream");
  }

  return ret;
}

static PyMemberDef brotli_Decompressor_members[] = {
  {NULL}  /* Sentinel */
};

static PyMethodDef brotli_Decompressor_methods[] = {
  {"process", (PyCFunction)brotli_Decompressor_process, METH_VARARGS, brotli_Decompressor_process_doc},
  {"is_finished", (PyCFunction)brotli_Decompressor_is_finished, METH_NOARGS, brotli_Decompressor_is_finished_doc},
  {NULL}  /* Sentinel */
};

static PyTypeObject brotli_DecompressorType = {
  #if PY_MAJOR_VERSION >= 3
  PyVarObject_HEAD_INIT(NULL, 0)
  #else
  PyObject_HEAD_INIT(NULL)
  0,                                     /* ob_size*/
  #endif
  "brotli.Decompressor",                   /* tp_name */
  sizeof(brotli_Decompressor),             /* tp_basicsize */
  0,                                       /* tp_itemsize */
  (destructor)brotli_Decompressor_dealloc, /* tp_dealloc */
  0,                                       /* tp_print */
  0,                                       /* tp_getattr */
  0,                                       /* tp_setattr */
  0,                                       /* tp_compare */
  0,                                       /* tp_repr */
  0,                                       /* tp_as_number */
  0,                                       /* tp_as_sequence */
  0,                                       /* tp_as_mapping */
  0,                                       /* tp_hash  */
  0,                                       /* tp_call */
  0,                                       /* tp_str */
  0,                                       /* tp_getattro */
  0,                                       /* tp_setattro */
  0,                                       /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                      /* tp_flags */
  brotli_Decompressor_doc,                 /* tp_doc */
  0,                                       /* tp_traverse */
  0,                                       /* tp_clear */
  0,                                       /* tp_richcompare */
  0,                                       /* tp_weaklistoffset */
  0,                                       /* tp_iter */
  0,                                       /* tp_iternext */
  brotli_Decompressor_methods,             /* tp_methods */
  brotli_Decompressor_members,             /* tp_members */
  0,                                       /* tp_getset */
  0,                                       /* tp_base */
  0,                                       /* tp_dict */
  0,                                       /* tp_descr_get */
  0,                                       /* tp_descr_set */
  0,                                       /* tp_dictoffset */
  (initproc)brotli_Decompressor_init,      /* tp_init */
  0,                                       /* tp_alloc */
  brotli_Decompressor_new,                 /* tp_new */
};

PyDoc_STRVAR(brotli_decompress__doc__,
"Decompress a compressed byte string.\n"
"\n"
"Signature:\n"
"  decompress(string)\n"
"\n"
"Args:\n"
"  string (bytes): The compressed input data.\n"
"\n"
"Returns:\n"
"  The decompressed byte string.\n"
"\n"
"Raises:\n"
"  brotli.error: If decompressor fails.\n");

static PyObject* brotli_decompress(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject *ret = NULL;
  Py_buffer input;
  const uint8_t* next_in;
  size_t available_in;
  int ok;

  static const char *kwlist[] = {"string", NULL};

#if PY_MAJOR_VERSION >= 3
  ok = PyArg_ParseTupleAndKeywords(args, keywds, "y*|:decompress",
                                   const_cast<char **>(kwlist), &input);
#else
  ok = PyArg_ParseTupleAndKeywords(args, keywds, "s*|:decompress",
                                   const_cast<char **>(kwlist), &input);
#endif

  if (!ok)
    return NULL;

  std::vector<uint8_t> output;

  /* >>> Pure C block; release python GIL. */
  Py_BEGIN_ALLOW_THREADS

  BrotliDecoderState* state = BrotliDecoderCreateInstance(0, 0, 0);

  BrotliDecoderResult result = BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT;
  next_in = static_cast<uint8_t*>(input.buf);
  available_in = input.len;
  while (result == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
    size_t available_out = 0;
    result = BrotliDecoderDecompressStream(state, &available_in, &next_in,
                                           &available_out, 0, 0);
    const uint8_t* next_out = BrotliDecoderTakeOutput(state, &available_out);
    if (available_out != 0)
      output.insert(output.end(), next_out, next_out + available_out);
  }
  ok = result == BROTLI_DECODER_RESULT_SUCCESS && !available_in;
  BrotliDecoderDestroyInstance(state);

  Py_END_ALLOW_THREADS
  /* <<< Pure C block end. Python GIL reacquired. */

  PyBuffer_Release(&input);
  if (ok) {
    ret = PyBytes_FromStringAndSize((char*)(output.size() ? &output[0] : NULL), output.size());
  } else {
    PyErr_SetString(BrotliError, "BrotliDecompress failed");
  }

  return ret;
}

static PyMethodDef brotli_methods[] = {
  {"decompress", (PyCFunction)brotli_decompress, METH_VARARGS | METH_KEYWORDS, brotli_decompress__doc__},
  {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(brotli_doc, "Implementation module for the Brotli library.");

#if PY_MAJOR_VERSION >= 3
#define INIT_BROTLI   PyInit__brotli
#define CREATE_BROTLI PyModule_Create(&brotli_module)
#define RETURN_BROTLI return m
#define RETURN_NULL return NULL

static struct PyModuleDef brotli_module = {
  PyModuleDef_HEAD_INIT,
  "_brotli",
  brotli_doc,
  0,
  brotli_methods,
  NULL,
  NULL,
  NULL
};
#else
#define INIT_BROTLI   init_brotli
#define CREATE_BROTLI Py_InitModule3("_brotli", brotli_methods, brotli_doc)
#define RETURN_BROTLI return
#define RETURN_NULL return
#endif

PyMODINIT_FUNC INIT_BROTLI(void) {
  PyObject *m = CREATE_BROTLI;

  BrotliError = PyErr_NewException((char*) "brotli.error", NULL, NULL);
  if (BrotliError != NULL) {
    Py_INCREF(BrotliError);
    PyModule_AddObject(m, "error", BrotliError);
  }

  if (PyType_Ready(&brotli_CompressorType) < 0) {
    RETURN_NULL;
  }
  Py_INCREF(&brotli_CompressorType);
  PyModule_AddObject(m, "Compressor", (PyObject *)&brotli_CompressorType);

  if (PyType_Ready(&brotli_DecompressorType) < 0) {
    RETURN_NULL;
  }
  Py_INCREF(&brotli_DecompressorType);
  PyModule_AddObject(m, "Decompressor", (PyObject *)&brotli_DecompressorType);

  PyModule_AddIntConstant(m, "MODE_GENERIC", (int) BROTLI_MODE_GENERIC);
  PyModule_AddIntConstant(m, "MODE_TEXT", (int) BROTLI_MODE_TEXT);
  PyModule_AddIntConstant(m, "MODE_FONT", (int) BROTLI_MODE_FONT);

  char version[16];
  snprintf(version, sizeof(version), "%d.%d.%d",
      BROTLI_VERSION >> 24, (BROTLI_VERSION >> 12) & 0xFFF, BROTLI_VERSION & 0xFFF);
  PyModule_AddStringConstant(m, "__version__", version);

  RETURN_BROTLI;
}
