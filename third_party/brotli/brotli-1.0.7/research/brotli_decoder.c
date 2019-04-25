/* Copyright 2018 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <brotli/decode.h>

#define BUFFER_SIZE (1u << 20)

typedef struct Context {
  FILE* fin;
  FILE* fout;
  uint8_t* input_buffer;
  uint8_t* output_buffer;
  BrotliDecoderState* decoder;
} Context;

void init(Context* ctx) {
  ctx->fin = 0;
  ctx->fout = 0;
  ctx->input_buffer = 0;
  ctx->output_buffer = 0;
  ctx->decoder = 0;
}

void cleanup(Context* ctx) {
  if (ctx->decoder) BrotliDecoderDestroyInstance(ctx->decoder);
  if (ctx->output_buffer) free(ctx->output_buffer);
  if (ctx->input_buffer) free(ctx->input_buffer);
  if (ctx->fout) fclose(ctx->fout);
  if (ctx->fin) fclose(ctx->fin);
}

void fail(Context* ctx, const char* message) {
  fprintf(stderr, "%s\n", message);
  exit(1);
}

int main(int argc, char** argv) {
  Context ctx;
  BrotliDecoderResult result = BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT;
  size_t available_in;
  const uint8_t* next_in;
  size_t available_out = BUFFER_SIZE;
  uint8_t* next_out;
  init(&ctx);

  ctx.fin = fdopen(STDIN_FILENO, "rb");
  if (!ctx.fin) fail(&ctx, "can't open input file");
  ctx.fout = fdopen(STDOUT_FILENO, "wb");
  if (!ctx.fout) fail(&ctx, "can't open output file");
  ctx.input_buffer = (uint8_t*)malloc(BUFFER_SIZE);
  if (!ctx.input_buffer) fail(&ctx, "out of memory / input buffer");
  ctx.output_buffer = (uint8_t*)malloc(BUFFER_SIZE);
  if (!ctx.output_buffer) fail(&ctx, "out of memory / output buffer");
  ctx.decoder = BrotliDecoderCreateInstance(0, 0, 0);
  if (!ctx.decoder) fail(&ctx, "out of memory / decoder");
  BrotliDecoderSetParameter(ctx.decoder, BROTLI_DECODER_PARAM_LARGE_WINDOW, 1);

  next_out = ctx.output_buffer;
  while (1) {
    if (result == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT) {
      if (feof(ctx.fin)) break;
      available_in = fread(ctx.input_buffer, 1, BUFFER_SIZE, ctx.fin);
      next_in = ctx.input_buffer;
      if (ferror(ctx.fin)) break;
    } else if (result == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
      fwrite(ctx.output_buffer, 1, BUFFER_SIZE, ctx.fout);
      if (ferror(ctx.fout)) break;
      available_out = BUFFER_SIZE;
      next_out = ctx.output_buffer;
    } else {
      break;
    }
    result = BrotliDecoderDecompressStream(
        ctx.decoder, &available_in, &next_in, &available_out, &next_out, 0);
  }
  if (next_out != ctx.output_buffer) {
    fwrite(ctx.output_buffer, 1, next_out - ctx.output_buffer, ctx.fout);
  }
  if ((result == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) || ferror(ctx.fout)) {
    fail(&ctx, "failed to write output");
  } else if (result != BROTLI_DECODER_RESULT_SUCCESS) {
    fail(&ctx, "corrupt input");
  }
  cleanup(&ctx);
  return 0;
}
